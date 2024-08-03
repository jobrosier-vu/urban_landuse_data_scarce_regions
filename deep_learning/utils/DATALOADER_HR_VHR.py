# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:14:47 2023

@author: JJR226
"""
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import pandas as pd
from collections import Counter
import itertools
import operator
from PIL import Image
from torchvision import transforms

def trim(data, delta):
    """Trims elements within `delta` of other elements in the list."""

    output = []
    last = 0

    for element in data:
        if element['value'] > last * (1 + delta):
            output.append(element)
            last = element['value']

    return output

def merge_lists(m, n):
    """
    Merges two lists into one.

    We do *not* remove duplicates, since we'd like to see all possible
    item combinations for the given approximate subset sum instead of simply
    confirming that there exists a subset that satisfies the given conditions.

    """
    merged = itertools.chain(m, n)
    return sorted(merged, key=operator.itemgetter('value'))


def approximate_subset_sum(data, target, epsilon):
    """
    Calculates the approximate subset sum total in addition
    to the items that were used to construct the subset sum.

    Modified to track the elements that make up the partial
    sums to then identify which subset items were chosen
    for the solution.

    """

    # Intialize our accumulator with the trivial solution
    acc = [{'value': 0, 'partials': [0]}]

    count = len(data)

    # Prep data by turning it into a list of hashes
    data = [{'value': d, 'partials': [d]} for d in data]

    for key, element in enumerate(data, start=1):
        augmented_list = [{
            'value': element['value'] + a['value'],
            'partials': a['partials'] + [element['value']]
        } for a in acc]

        acc = merge_lists(acc, augmented_list)
        acc = trim(acc, delta=float(epsilon) / (2 * count))
        acc = [val for val in acc if val['value'] <= target]

    # The resulting list is in ascending order of partial sums; the
    # best subset will be the last one in the list.
    return acc[-1]

def dataset_split(sample_list, target_list,split_fraction):
    """
    splits dataset in two with a split ratio given as input
    
    The ApproximateSubsetSums is used to get an split by summing the samples untill the split size is (approximately) reached
    per category
    
    output: indices of the samples regions to use in each split
    """
    zone_target_df = pd.DataFrame({'samples':sample_list,'targets':target_list}) 
    

    total_selected_indices_list = []
    len_previous_category = 0
    for category in  np.unique(zone_target_df.targets):
        zone_list = []
        for sample in zone_target_df[zone_target_df['targets']==category].samples:
            zone_list.append(int(sample[0].split('_')[-1][:-4]))
               
        # get the number of samples per zone
        count_dict = dict(Counter(zone_list).items())
        count_dict_values = list(count_dict.values())
        zone_name = list(count_dict.keys())
        split_size = round(len(zone_list)*split_fraction)
        
        # apply the approximate_subset_sum algorithm
        selected_amounts = approximate_subset_sum(count_dict_values, split_size, 0.1)
        amounts_array = np.array(selected_amounts['partials'])
        
        # based on the approximate_subset_sum algorithm get the zones of the samples that are selected
        selected_zone_list = []
        indices_used = []
        for i in amounts_array[1:]:
            getindex = np.where(count_dict_values==i)[0]
            getindex_filtered = [elem for elem in getindex if elem not in indices_used ]
            indices_used.append(getindex_filtered[0])
            selected_zone_list.append(zone_name[getindex_filtered[0]])
            
            
        
        # temp dataframe
        df = pd.DataFrame({'zones':zone_list})
        
        # get indices of samples that are within the selected zones
        indices_per_category = np.array(df.index[df.zones.isin(selected_zone_list)])
        
        # add tot the total list and correct for the size of the previous category
        total_selected_indices_list.extend(indices_per_category+len_previous_category)
        
        # update the previous size
        len_previous_category += len(zone_list)
    
    # get the other split of size (1-split_fraction)*total_length
    not_selected_indices = list(set(np.arange(0,len_previous_category)) - set(total_selected_indices_list))
    
    return total_selected_indices_list,not_selected_indices

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def rand_flip(img1):
    r = random.random()

    if r < 0.25:
        return img1
    elif r < 0.5:
        return torch.flip(img1,[1])
    elif r < 0.75:
        return torch.flip(img1,[2])
    else:
        return torch.flip(img1,[1,2])

def normalize(im1, mean, std):
    return (im1-mean)/std

def scale(im1, factor):
    return im1/factor

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for sample, _ in iter(dataloader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.nanmean(sample, [0,2,3])
        channels_squared_sum += torch.nanmean(torch.square(sample), [0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    # return mean and std as tensors with shape [3,1,1]
    temp = torch.zeros([sample.size()[1],1,1])
    temp[:,0,0] = torch.tensor(mean)
    tenmean = temp.detach().clone()
    temp[:,0,0] = torch.tensor(std)
    tensstd = temp.detach().clone()
    return tenmean, tensstd

class Subset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        
        if 'augment' in self.transform:
            im = rand_flip(im)  
        
        if 'normalize' in self.transform:
            im = normalize(im, self.transform['normalize'][0], self.transform['normalize'][1])
            
        if 'scale' in self.transform:
            im = scale(im, self.transform['scale'][0])
        
        return im, labels

    def __len__(self):
        return len(self.indices)
        
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def transform_to_tensor(im):
    convert_tensor = transforms.ToTensor()
    return convert_tensor(im)

def im_loader(path):
    sample = Image.open(path) 
    return transform_to_tensor(sample)  
    
def sample_dataset_uniform(input_dataset,number_samples):
    
    traindataset = input_dataset.dataset
    transform = traindataset.transform
    
    train_array = np.array(traindataset.dataset.samples)[traindataset.indices]
    total_array = np.array(traindataset.dataset.samples)
    
    sample_df = pd.DataFrame({'samples':train_array[:,0],'target':train_array[:,1]})
    sample_df.set_index=traindataset.indices
    
    
    samples_selected=[]
    for target in np.unique(sample_df.target):
        sample_df_pc = sample_df[sample_df['target']==target]
        n = round((len(sample_df_pc))/number_samples+1)
        nthsample = sample_df_pc.iloc[::n, :]
    
        samples_selected.extend(nthsample.samples)
        
    itemindex = [np.where(total_array==i)[0][0] for i in samples_selected]
    selected_dataset = Subset(traindataset.dataset, itemindex, transform)
    
    train_subset_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=10 ,num_workers=0, shuffle=True)
    return train_subset_loader

def sample_dataset_random(input_dataset,number_samples):
    
    traindataset = input_dataset.dataset
    transform = traindataset.transform
    
    train_array = np.array(traindataset.dataset.samples)[traindataset.indices]
    total_array = np.array(traindataset.dataset.samples)
    
    sample_df = pd.DataFrame({'samples':train_array[:,0],'target':train_array[:,1]})
    sample_df.set_index=traindataset.indices
    
    samples_selected=[]
    for target in np.unique(sample_df.target):
        sample_df_pc = sample_df[sample_df['target']==target]
        selected = random.sample(list(sample_df_pc.samples),number_samples)
    
        samples_selected.extend(selected)
        
    itemindex = [np.where(total_array==i)[0][0] for i in samples_selected]
    selected_dataset = Subset(traindataset.dataset, itemindex, transform)
    
    train_subset_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=25 ,num_workers=2, shuffle=True)
    return train_subset_loader

def Dataloader_train_val_test(path,val_test_size, num_workers,batch_size,transform,input_type,mode):
    if input_type == 'numpy':
        dataset = torchvision.datasets.DatasetFolder(
        root=path,
        loader=npy_loader,
        extensions=('.npy')
        )
    if input_type=='image':
        dataset = torchvision.datasets.DatasetFolder(
        root=path,
        loader=im_loader,
        extensions=('.jpg')
        )        
    
    # split the dataset in main and minor part
    main,minor = dataset_split(dataset.samples, dataset.targets,1-(val_test_size[0]+val_test_size[1]) )
    
    # get a list of samples and targets that are used for testing and validation
    minor.sort()
    TestValSamples = [dataset.samples[i] for i in minor]
    TestValTargets = [dataset.targets[i] for i in minor]
    
    # split the samples in a validation and a test set
    test_indices_list,val_selected_indices = dataset_split(TestValSamples, TestValTargets, val_test_size[1]/(val_test_size[0]+val_test_size[1]))
    Test_indices = [minor[i] for i in test_indices_list]
    Val_indices = [minor[i] for i in val_selected_indices]
    
    if mode=='test':
        #extend the indices of train with the indices of val
        main.extend(Val_indices)
    
    #get normalization stats from training data 
    if transform==None:
        train_dataset = Subset(dataset,main,[])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=False)
        mean, std = get_mean_and_std(train_loader)
        transform = {'augment':'rand_flip','normalize':[mean,std]}
    
    # use indices to create a subset of the dataset
    train_dataset = Subset(dataset,main,transform)    
    
    
    # use indices to create a subset of the dataset remove augment for testing and validation
    transfrom_val_test = transform.copy()
    transfrom_val_test.pop('augment', None)
    test_dataset = Subset(dataset,Test_indices,transfrom_val_test)
    val_dataset = Subset(dataset,Val_indices,transfrom_val_test)
    
      
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=False)
        

    return train_loader,val_loader,test_loader


def Dataloader_pretrained(path,test_size, num_workers,batch_size,transform,input_type,num_samples):
    if input_type == 'numpy':
        dataset = torchvision.datasets.DatasetFolder(
        root=path,
        loader=npy_loader,
        extensions=('.npy')
        )
    if input_type=='image':
        dataset = torchvision.datasets.DatasetFolder(
        root=path,
        loader=im_loader,
        extensions=('.jpg')
        )        
    
    # split the dataset in main and minor part
    main,minor = dataset_split(dataset.samples, dataset.targets,1-test_size )
    
    #get normalization stats from training data 
    if transform==None:
        train_dataset = Subset(dataset,main,[])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=True)
        mean, std = get_mean_and_std(train_loader)
        transform = {'augment':'rand_flip','normalize':[mean,std]}
    
    # use indices to create a subset of the dataset
    train_dataset = Subset(dataset,main,transform)
    
    # get a list of samples and targets that are used for testing and validation
    minor.sort()
    TestSamples = [dataset.samples[i] for i in minor]
    TestTargets = [dataset.targets[i] for i in minor]
    
    # use indices to create a subset of the dataset remove augment for testing and validation
    transform.pop('augment', None)
    test_dataset = Subset(dataset,minor,transform)
    
    # use the selection function to select n samples from dataset
    selected_trainset = select_samples(train_dataset, num_samples)
    
    # create the dataloaders        
    train_loader = torch.utils.data.DataLoader(selected_trainset, batch_size=batch_size ,num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=False)
    
    return train_loader,test_loader

###### test
if __name__ == "__main__":
    path = r''
    val_test_size = 0.2
    batch_size = 30
    num_workers = 0
    #transform = {'augment':'rand_flip','normalize':[torch.tensor([[[747.9951]],[[ 918.8052]],[[ 1106.4594]],[[ 2106.8010]]]),torch.tensor([[[392.6980]],[[353.6538]],[[410.5128]],[[444.0011]]])]}
    transform = {'augment':'rand_flip','scale':[torch.tensor([[[255]],[[ 255]],[[ 255]]])]}
    input_type='image'
    train_loader,val_loader,test_loader = Dataloader_train_val_test(path,val_test_size, num_workers,batch_size,transform,input_type)
    
