import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet101
from tqdm import tqdm
import random
import numpy as np
from easyfsl.methods import MatchingNetworks, RelationNetworks
from statistics import mean
from easyfsl.samplers import TaskSampler
import torch.utils.model_zoo
import copy
from pathlib import Path
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
from easyfsl.methods import FewShotClassifier
from typing import Optional, Tuple
from torch import Tensor
import random

# ensure a deterministic process for reproducibility
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda"

cnn = resnet101(pretrained=True)

# set the fc classifier into an identity layer
cnn.fc = nn.Identity()    

# enclosing cnn into matching network
# output feature dimension from the resnet is 2048
fsc = MatchingNetworks(cnn, feature_dimension=2048).to(DEVICE) 

# load the dataset to feed it into the datalaoder
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.file_paths = self.get_file_paths()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        img_path, label = self.file_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_file_paths(self):
        file_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                file_paths.append((img_path, class_idx))

        return file_paths

# transformations
transform = transforms.Compose([
    transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(root_dir='surrogate/train', transform=transform)
val_dataset = CustomDataset(root_dir='surrogate/validation', transform=transform)
test_dataset = CustomDataset(root_dir='surrogate/testing', transform=transform)

n_way = 5
n_shot = 5
n_query = 5
n_tasks_per_epoch = 500
n_validation_tasks = 100

train_dataset.get_labels = lambda: [instance[1] for instance in train_dataset]
val_dataset.get_labels = lambda: [instance[1] for instance in val_dataset]
test_dataset.get_labels = lambda: [instance[1] for instance in test_dataset]

# special batch samplers that sample few-shot classification tasks with a pre-defined shape
train_sampler = TaskSampler(
    train_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch
)
val_sampler = TaskSampler(
    val_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
)
test_sampler = TaskSampler(
    test_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
)

# dataloading with a customized collate_fn so that batches are delivered
# in the shape: (support_images, support_labels, query_images, query_labels, class_ids)
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)
test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

LOSS_FUNCTION = nn.CrossEntropyLoss() 

n_epochs = 400
scheduler_milestones = [250, 350]
scheduler_gamma = 0.1
learning_rate = 1e-2
tb_logs_dir = Path(".")

train_optimizer = SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
train_scheduler = MultiStepLR(train_optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)

# set up tensorboard writer to log training and validation loss
tb_writer = SummaryWriter("tensorboard_logs/MatchingNet_S")

# runs the cnn for one epoch 
def training_epoch(model, data_loader, optimizer):
    all_loss = []
    model.train()
    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for _, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(support_images.to(DEVICE), support_labels.to(DEVICE))
            classification_scores = model(query_images.to(DEVICE))

            loss = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            all_loss.append(loss.item())
            
            tqdm_train.set_postfix(loss=mean(all_loss))
    return mean(all_loss)

# modified evaluation function for tracking validation loss
# obtained EasyFSL (Bennequin, n.d.) 
def evaluate_on_one_task(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
) -> Tuple[int, int]:
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data
    loss = LOSS_FUNCTION(predictions, query_labels)
    number_of_correct_predictions = (
        (torch.max(predictions, 1)[1] == query_labels).sum().item()
    )
    return number_of_correct_predictions, len(query_labels), loss

# modified evaluation function for tracking validation loss
# obtained EasyFSL (Bennequin, n.d.) 
def evaluate(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> float:

    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    all_loss = []

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.eval()
    with torch.no_grad():
        # We use a tqdm context to show a progress bar in the logs
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                correct, total, loss = evaluate_on_one_task(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                )
                
                total_predictions += total
                correct_predictions += correct
                all_loss.append(loss.item())

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    return correct_predictions / total_predictions, mean(all_loss)

best_validation_accuracy = 0.0
validation_frequency = 1
# train loop
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    average_loss = training_epoch(fsc, train_loader, train_optimizer)
        
    validation_accuracy, validation_loss = evaluate(fsc, val_loader, tqdm_prefix="Validation")

    if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            print("Best model")
            
    tb_writer.add_scalar("Validation_loss", validation_loss, epoch)
        
    tb_writer.add_scalar("Train_loss", average_loss, epoch)
    train_scheduler.step()

# evaluate model 
test_accuracy = evaluate(fsc, test_loader, tqdm_prefix="Testing")

# save model
torch.save(fsc, "best_models/MatchingNet_S")
