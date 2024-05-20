import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from easyfsl.samplers import TaskSampler
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from statistics import mean
from torchvision.models import resnet101
from copy import deepcopy
import random

# ensure a deterministic process for reproducibility
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cuda'

cnn = resnet101(pretrained=True)

# freezing all layers below the first residual block
p = list(cnn.parameters())
for param in p[33:]:
    param.requires_grad = False

# add fully connected classifier 
# out_features is set to 5 because the FSL sampler re-labels all images 
# according to n-way (this case, 5-way)
cnn.fc = nn.Sequential(
    nn.Linear(in_features=cnn.fc.in_features, out_features=4096), 
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=4096, out_features=4096),  
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=4096, out_features=5), 
)

cnn.to(DEVICE)

class Reptile:
    """
    task_udate: run n steps of gradient descent on the support set. If train=True, 
    it also run n steps on the query set. train=False is for evaluation purposes.

    meta_update: updates the parameters of the model using the previous parameters and
    the task-specific parameters obtained after task_update.

    training_epoch: runs Reptile fro one epoch. 

    evaluate_on_one_task: outputs the prediction of the backbone and the loss.

    evaluate: evaluates the model by running task_update with train=False to 
    expose the model to the support set and then run evaluate_on_one_task to obtain
    the prediction in the query set.
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.01, num_updates=5):
        self.model = model
        self.meta_lr = meta_lr
        self.num_updates = num_updates
        self.optimizer = SGD(cnn.parameters(), lr=inner_lr, momentum=0.9, weight_decay=1e-5)
        self.task_params = None
        self.all_loss = []
        self.current_params = deepcopy(self.model.state_dict())
    
    def task_update(self, x_support, y_support, x_query, y_query, train=True):
        self.all_loss = []
        # Perform inner updates on the model using support set
        for _ in range(self.num_updates):
            self.optimizer.zero_grad()
            y_pred = self.model(x_support)
            loss = LOSS_FUNCTION(y_pred, y_support)
            loss.backward()
            self.optimizer.step()

            if train:
                # Update the model parameters based on the query set
                self.optimizer.zero_grad()
                y_pred = self.model(x_query)
                loss = LOSS_FUNCTION(y_pred, y_query)
                loss.backward()
                self.optimizer.step()
                self.all_loss.append(loss.item())

                self.task_params = deepcopy(self.model.state_dict())

    def meta_update(self):
        self.model.load_state_dict({name : 
        self.current_params[name] + (self.task_params[name] - self.current_params[name]) * self.meta_lr 
        for name in self.current_params})

        self.current_params = deepcopy(self.model.state_dict())

    def training_epoch(self, data_loader):
        self.model.train()
        with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
            for _, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
                
                support_images, support_labels, query_images, query_labels = support_images.to(DEVICE), support_labels.to(DEVICE), query_images.to(DEVICE), query_labels.to(DEVICE)

                fsc.task_update(support_images, support_labels, query_images, query_labels)
                tqdm_train.set_postfix(loss=mean(self.all_loss))
                fsc.meta_update()

        return mean(self.all_loss)

    def evaluate_one_task(self, x_query, y_query):
        y_pred = self.model(x_query)
        loss = LOSS_FUNCTION(y_pred, y_query)
        pred = y_pred.data.max(1, keepdim=True)[1]
        return pred.eq(y_query.data.view_as(pred)).sum().item(), len(y_query), loss
    
    def evaluate(self, dataloader, tqdm_prefix):
        total_predictions = 0
        correct_predictions = 0
        val_loss = []
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=tqdm_prefix) as tqdm_eval:
                for _, (support_images, support_labels, query_images, query_labels, _) in tqdm_eval:
                    
                    support_images, support_labels, query_images, query_labels = support_images.to(DEVICE), support_labels.to(DEVICE), query_images.to(DEVICE), query_labels.to(DEVICE)
                    
                    # enable train() to perform gradient steps on the support set
                    self.model.train()
                    fsc.task_update(support_images, support_labels, query_images, query_labels, train=False)

                    # set to eval()	to evaluate on query set
                    self.model.eval()
                    correct, total, validation_loss = fsc.evaluate_one_task(query_images, query_labels)

                    total_predictions += total
                    correct_predictions += correct
                    
                    val_loss.append(validation_loss.item())

		            # discard the task adapted parameters for the meta-training parameters
                    self.model.load_state_dict(self.current_params)

                    # Log accuracy in real time
                    tqdm_eval.set_postfix(accuracy=(correct_predictions/total_predictions))

        return correct_predictions / total_predictions, mean(val_loss)

fsc = Reptile(cnn)

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

n_epochs = 200
scheduler_milestones = [150]
scheduler_gamma = 0.1
tb_logs_dir = Path(".")

train_scheduler = MultiStepLR(fsc.optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)

# set up tensorboard writer to log training and validation loss
tb_writer = SummaryWriter("tensorboard_logs/Reptile_P")

# training loop
best_validation_accuracy = 0.0
validation_frequency = 1
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    average_loss = fsc.training_epoch(train_loader)

    if epoch % validation_frequency == validation_frequency - 1:
    
        validation_accuracy, validation_loss = fsc.evaluate(val_loader, tqdm_prefix="Validation")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            print("Best model")
            
        tb_writer.add_scalar("Validation_loss", validation_loss, epoch)
        
    tb_writer.add_scalar("Train_loss", average_loss, epoch)
    train_scheduler.step()

# evaluate on test set
test_accuracy = fsc.evaluate(test_loader, tqdm_prefix="Testing")
print(test_accuracy)

# save model
torch.save(fsc.model, "best_models/Reptile_P")

