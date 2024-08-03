# -*- coding: utf-8 -*-
"""
code for densenet from: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html#DenseNet

@author: botan
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

import warnings
warnings.simplefilter("ignore")

# set base path
BASEPATH = r'C:\Users\JJR226\Documents\PhD\Paper3\DL_verified'

sys.path.append(BASEPATH)
from utils.DATALOADER_HR_VHR_BF_150 import *
from models.Densenet42_BF_seperate import *
from utils.plotting_CM import *
from utils.combine_categories import *

def log_additional_input(direc,dictionary,log_dataset):
    with open(os.path.join(direc,'input_params.txt'), 'w') as f:
        for key_value in zip(dictionary.keys(),dictionary.values()):
            print(key_value, file=f)
            
    with open(os.path.join(direc,'samples.txt'), 'w') as f:
        for sample_name in log_dataset.dataset.dataset.samples:
            print(sample_name, file=f) 

class DENSEModule(pl.LightningModule):
    def __init__(self, model, model_hparams, optimizer_name, optimizer_hparams, loss_weights,alpha):
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
        self.loss_module = nn.CrossEntropyLoss(weight=loss_weights.float(),reduction='mean')
        # Example input for visualizing the graph in Tensorboard
        self.predictions = []
        # store the targets
        self.test_y = []
        self.alpha = alpha
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=3 )
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler ,
        'monitor': 'val_loss',
    }  

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss1 = self.loss_module(preds[0], labels)
        loss2 = self.loss_module(preds[1], labels)
        
        alpha1 = 1-self.alpha
        summend_preds = torch.sum(torch.stack([preds[0]*alpha1,preds[1]*self.alpha]), dim=0) 
        loss = loss1*alpha1 + loss2*self.alpha
        acc = (summend_preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        self.log("train_loss1", loss1)
        self.log("train_loss2", loss2)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss1 = self.loss_module(preds[0], labels)
        loss2 = self.loss_module(preds[1], labels)
        
        alpha1 = 1-self.alpha
        summend_preds = torch.sum(torch.stack([preds[0]*alpha1,preds[1]*self.alpha]), dim=0)
        loss = loss1*alpha1 + loss2*self.alpha
        acc = (summend_preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        
        alpha1 = 1-self.alpha
        summend_preds = torch.sum(torch.stack([preds[0]*alpha1,preds[1]*self.alpha]), dim=0).argmax(dim=-1)
        acc = (summend_preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        y_batch = labels.cpu().numpy()
        self.test_y.extend(y_batch)
        
        pred_cpu = summend_preds.cpu().numpy()
        self.predictions.extend(pred_cpu)
        
        self.log("test_acc", acc)
        

def train_model(model_name,model_hparams,optimizer_name,optimizer_hparams,weights,alpha,save_name,version_id,data_loaders,pretrained_filename,input_dict, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    
    logger = TensorBoardLogger(save_name, version=version_id)
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        logger=logger,  # Where to save models
        # We run on a single GPU (if possible)
        gpus=1 if str(device) == "cuda:0" else 0,
        # How many epochs to train for if no patience is set
        max_epochs=15,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc", save_last=True),            
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            #LearningRateMonitor("epoch"),EarlyStopping(monitor="val_loss", mode="min", patience=15)
        ],  # Log learning rate every epoch
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    
   
    # Check whether pretrained model exists. If yes, load it
    if pretrained_filename:
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = DENSEModule.load_from_checkpoint(pretrained_filename)
        
    else:        
        model = DENSEModule(model_name, model_hparams,optimizer_name,optimizer_hparams,weights,alpha)
    
    # train the model
    trainer.fit(model, data_loaders[0], data_loaders[1])
        
    # run on validation data and plot the confusion matrix
    trainer.test(model=model, dataloaders=data_loaders[1])
    y_pred = model.predictions
    y_true = model.test_y  
      
    plot_cm(y_true, y_pred,logger.log_dir, version_id,data_loaders[1].dataset.dataset.classes, figsize=(8,8),dpi=100)
      
    # Add the input parameters to the logging directory and the samples of the test set to verify there is no overlap bewteen train and test data
    log_additional_input(logger.log_dir,input_dict,data_loaders[2])
    
    

if __name__ == "__main__":
    
    input_variables = ['DenseNet42_Kampala_Planet_150px_30sp_3res_Planet_BF300_lr2_SGD_comb2_weights_0_test_02_pretrained_samplesize_10.npy']
    setting = 'random_datasampler_val_epochs15_patience3'
    for input_file in input_variables:
        read_dictionary = np.load(os.path.join(BASEPATH,'training','INPUT_VALUES',input_file),allow_pickle='TRUE').item()

        # experiment parameters
        model_name = read_dictionary['model_name']
        dataset_name = read_dictionary['dataset_name']
        building_footprint = read_dictionary['building_footprint']
        experiment_name = read_dictionary['experiment_name']
    
        # default no combinations of categories are made
        combinations = read_dictionary['combinations']
        alpha = read_dictionary['alpha']
        
        # dataset parameters
        val_test_size = read_dictionary['val_test_size']
        batch_size = read_dictionary['batch_size']
        num_workers = read_dictionary['num_workers']
        
        # hyperparameters
        optimizer = read_dictionary['optimizer']
        learning_rate = read_dictionary['learning_rate']
        weight_increase = read_dictionary['weight_increase']
    
        # name of the pretrained file if applicable
        pretrained_network = read_dictionary['pretrained_network']
        
        #sample size is applicable
        sample_size = read_dictionary['sample_size']      
        
        # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
        DATASET_PATH = os.environ.get(r"data", os.path.join(BASEPATH,'data',dataset_name))
        # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
        BF_DATASET_PATH = os.environ.get(r"data", os.path.join(BASEPATH,'data',"BF_{}".format(dataset_name),building_footprint))    
        # Path to the folder where the pretrained models are saved
        CHECKPOINT_PATH = os.environ.get(r"Checkpoints", os.path.join(BASEPATH,'Checkpoints'))
        # name the model checkpoint based on the dataset used
        SAVE_PATH = os.path.join(r'C:\Users\JJR226\Desktop\checkpoints',dataset_name,model_name,experiment_name, setting)
        # Create checkpoint path if it doesn't exist yet
        os.makedirs(SAVE_PATH, exist_ok=True)    
    
        # create dataset and check input type
        if dataset_name.split('_')[1]=='Google':
            num_input_layers = 3
            num_layers = [6, 12]
            growth_rate = 32
            bin_size = [31,15,7]
            lin_size = 5376
            extension =  ('.jpg')
            transform = {'augment':'rand_flip','scale':[torch.tensor([[[255]],[[ 255]],[[ 255]]])]}
        elif dataset_name.split('_')[1]=='Planet':
            input_type='numpy'
            num_input_layers = 4
            num_layers = [6, 12]
            growth_rate = 32
            bin_size = [6,3,2]
            lin_size = 3584
            extension =  ('.npy')
            transform = []
            

        for seed in [0,1,2,3,4,5,6,7,8,9]:


            # specify the version based on the seed
            version_id = 'seed_{}'.format(seed)
    
            # Function for setting the seed
            pl.seed_everything(seed)
                     
            # set up the dataloaders    
            train_loader,val_loader,test_loader = Dataloader_train_val_test(DATASET_PATH,BF_DATASET_PATH,extension,val_test_size,transform,batch_size,num_workers,'test')
               
            # apply any reclassification of categories
            [train_loader,val_loader,test_loader],new_categories = combine_categories([train_loader,val_loader,test_loader],combinations)
            
            # when training without pretrained network
            train_loader.dataset.transform = []
            
            if not sample_size == None:
                train_loader = sample_dataset_random(train_loader,sample_size)
                meanBF,stdBF = BF_stats(train_loader.dataset.dataset,train_loader.dataset.indices)
                mean, std = get_mean_and_std(train_loader)
                transform = {'augment':'rand_flip','normalize':[mean,std,meanBF,stdBF]}
                train_loader.dataset.transform = transform  
                transfrom_val_test = transform.copy()
                transfrom_val_test.pop('augment', None) 
                test_loader.dataset.transform = transform  
                
            # get class weights 
            class_sample_count = np.unique(np.array(train_loader.dataset.dataset.samples)[train_loader.dataset.indices][:,1], return_counts=True)[1]
            weight = len(np.array(train_loader.dataset.dataset.samples)[:,1]) / (len(train_loader.dataset.dataset.classes) * class_sample_count)
            
            # increase the weight of some classes
            samples_weight = torch.from_numpy(weight)/torch.from_numpy(weight)
            samples_weight[0] = samples_weight[0]+weight_increase  

            # Ensure that all operations are deterministic on GPU (if used) for reproducibility
            torch.backends.cudnn.determinstic = True
            torch.backends.cudnn.benchmark = False
            
            #select device
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            
            
            # specify model
            model_dict = {}
            act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}
            model = DenseNet_BF
            
            # get number of classes
            num_classes = len(np.unique(np.array(val_loader.dataset.dataset.samples)[:,1]))
            
            # train the model  
            train_model(model,
                        model_hparams={'num_input_layers':num_input_layers,'in_channels_BF':13,"num_classes": num_classes,'alpha':alpha,"num_layers": num_layers,"bn_size": 2,"growth_rate": growth_rate,"act_fn_name": "relu",'lin_size':lin_size,'bin_size':bin_size},
                        optimizer_name=optimizer,
                        optimizer_hparams={"lr": learning_rate, "weight_decay": 1e-4},
                        weights=samples_weight,
                        alpha=alpha,
                        save_name=SAVE_PATH,
                        version_id=version_id,
                        data_loaders = [train_loader,val_loader, test_loader],
                        pretrained_filename = pretrained_network,
                        input_dict = read_dictionary)