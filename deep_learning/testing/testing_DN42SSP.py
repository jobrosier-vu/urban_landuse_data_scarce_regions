# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 09:33:22 2023

@author: JJR226
"""
import os
import glob
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import sys
import json

# set base path
BASEPATH = r'C:\Users\JJR226\Documents\PhD\Paper3\DL_verified'

sys.path.append(BASEPATH)
from utils.DATALOADER_HR_VHR import *
from models.Densenet42 import  *
from utils.plotting_CM_testing import *
from utils.combine_categories import *


class DENSEModule(pl.LightningModule):
    def __init__(self, model, model_hparams, optimizer_name, optimizer_hparams, loss_weights):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model(**model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss(weight=loss_weights.float())
        # Example input for visualizing the graph in Tensorboard
        self.predictions = []
        # store the targets
        self.test_y = [] 

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=5 )
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler ,
        'monitor': 'val_loss',
    }  

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        y_batch = labels.cpu().numpy()
        self.test_y.extend(y_batch)
        
        pred_cpu = preds.cpu().numpy()
        self.predictions.extend(pred_cpu)
        
        self.log("test_acc", acc)
        

def test_model(model_name,save_name,version,data_loaders,pretrained_filename=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    logger = TensorBoardLogger(save_name, version=version)
    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        # We run on a single GPU (if possible)
        gpus=1 if str(device) == "cuda:0" else 0)

    
    model = DENSEModule.load_from_checkpoint(pretrained_filename)   
    # run on validation data and plot the confusion matrix
    trainer.test(model=model, dataloaders=data_loaders)
    y_pred = model.predictions
    y_true = model.test_y  
      
    plot_cm(y_true, y_pred,save_name, version,data_loaders.dataset.dataset.classes, figsize=(8,8),dpi=100)
      
def extract_params(checkpoint):
    if not os.path.exists(os.path.join(checkpoint,'dict.txt')):
        lines = []
        with open(os.path.join(checkpoint,'input_params.txt')) as f:
            for line in f:
               line = line.replace(",", ":",1)
               line = line.replace("(", "")
               line = line.replace(")", "")
               line = line.replace("'", '"')
               lines.append(line)
        
        with open(os.path.join(checkpoint,'dict.txt'), 'w') as my_file:
            my_file.write('{')
            for line in lines:
                new_line = line.replace("\n","") + ",\n"
                if 'pretrained_network' in line:
                    new_line = new_line.replace("None","")
                    new_line = new_line.replace(",",'"None"')
                my_file.write(new_line)
            my_file.write('}')
    
    with open(os.path.join(checkpoint,'dict.txt')) as f:
        data = f.read()
    js = json.loads(data) 
    return js


if __name__ == "__main__":
    
    checkpoint = r'C:\Users\JJR226\Documents\PhD\Paper3\DL_verified\checkpoints\Nairobi_Planet_150px_30sp_3res\DenseNet42\Planet_lr2_SGD_comb2_weight_0_valtest_0102\default\seed_1'
       
    read_dictionary = extract_params(checkpoint)
    
    # experiment parameters
    model_name = read_dictionary['model_name']
    dataset_name = read_dictionary['dataset_name']
    building_footprint = read_dictionary['building_footprint']
    experiment_name = read_dictionary['experiment_name'] 
    version_id = int(os.path.split(checkpoint)[1].split('_')[1])

    # default no combinations of categories are made
    combinations = read_dictionary['combinations']
    
    # dataset parameters
    val_test_size = read_dictionary['val_test_size']
    batch_size = read_dictionary['batch_size']
    num_workers = read_dictionary['num_workers']
           
    # name of the pretrained file if applicable
    pretrained_network = os.path.join(checkpoint,'checkpoints','last.ckpt')
    
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = os.environ.get(r"data", os.path.join(BASEPATH,'data',dataset_name))
        
    # Function for setting the seed
    pl.seed_everything(version_id)
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    #select device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # create dataset and check input type
    if dataset_name.split('_')[1]=='Google':
        input_type='image'
        num_input_layers = 3
        num_layers = [6, 12]
        growth_rate = 32
        bin_size = [31,15,7]
        lin_size = 5376
        transform = {'augment':'rand_flip','scale':[torch.tensor([[[255]],[[ 255]],[[ 255]]])]}
    elif dataset_name.split('_')[1]=='Planet':
        input_type='numpy'
        num_input_layers = 4
        num_layers = [6, 12]
        growth_rate = 32
        bin_size = [6,3,2]
        lin_size = 3584
        transform = None
        
    # set up the dataloaders    
    train_loader,test_loader = Dataloader_full_training(DATASET_PATH,0.2, num_workers,batch_size,transform,input_type)
       
    # apply any reclassification of categories
    [train_loader,test_loader],new_categories = combine_categories([train_loader,test_loader],combinations)
    
    # specify model
    model_dict = {}
    act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}
    model = DenseNet
    
    # get number of classes
    num_classes = len(np.unique(val_loader.dataset.dataset.targets))
    
    # train the model  
    test_model(model,
                save_name=checkpoint,
                version='seed_{}'.format(version_id),
                data_loaders = test_loader,
                pretrained_filename = pretrained_network)

    
    