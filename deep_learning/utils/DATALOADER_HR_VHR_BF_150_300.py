"""
Code for approximate_subset_sum from:  https://nerderati.com/bartering-for-beers-with-approximate-subset-sums/'

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
import os

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
        sample = sample[0]
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
        self.im_to_tensor = transforms.ToTensor()
        
    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        
        if 'augment' in self.transform:
            im[0] = rand_flip(im[0])  
        
        if 'normalize' in self.transform:
            im[0] = normalize(im[0], self.transform['normalize'][0], self.transform['normalize'][1])
            im[1] = normalize(im[1], self.transform['normalize'][2], self.transform['normalize'][3])
            im[2] = normalize(im[2], self.transform['normalize'][4], self.transform['normalize'][5])
            
        if 'scale' in self.transform:
            im[0] = scale(im[0], self.transform['scale'][0])
            im[1] = normalize(im[1], self.transform['scale'][1], self.transform['scale'][2])
            im[2] = normalize(im[2], self.transform['scale'][3], self.transform['scale'][4])
        
        return im, labels

    def __len__(self):
        return len(self.indices)
    
   
class Fullset_vhr(torchvision.datasets.DatasetFolder): 
    
    def __init__(self, root_dir,total_df,extensions=('.jpg'), transform=None):    
        super(Fullset_vhr, self).__init__(root_dir, transform, extensions)
        self.total_df = total_df
        self.im_to_tensor = transforms.ToTensor()

      
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path) 
        
        sample_name = os.path.split(path)[1][:-4]+'.npy'
        stats_150 = list(self.total_df[self.total_df.names==sample_name].stats150)[0]
        stats_300 = list(self.total_df[self.total_df.names==sample_name].stats300)[0]
        return [self.im_to_tensor(sample), torch.tensor(stats_150), torch.tensor(stats_300)], target
        
 
class Fullset_hr(torchvision.datasets.DatasetFolder): 
    
    def __init__(self, root_dir,total_df,extensions=('.npy'), transform=None):    
        super(Fullset_hr, self).__init__(root_dir, transform, extensions)
        self.total_df = total_df
        self.im_to_tensor = transforms.ToTensor()

      
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = torch.from_numpy(np.load(path))
        
        sample_name = os.path.split(path)[1][:-4]+'.npy'
        stats_150 = list(self.total_df[self.total_df.names==sample_name].stats150)[0]
        stats_300 = list(self.total_df[self.total_df.names==sample_name].stats300)[0]
        
        return [sample, torch.tensor(stats_150), torch.tensor(stats_300)], target    
 

def BF_stats(dataset,main):
    train_dataset = Subset(dataset,main,[])
    
    samples = pd.DataFrame.from_records(train_dataset.dataset.samples)[0]
    indices = train_dataset.indices
    
    selected_names = samples[samples.index.isin(indices)]
    sample_name = [os.path.split(name)[1][:-4]+'.npy' for name in selected_names]
    
    train_df_150 = train_dataset.dataset.total_df[train_dataset.dataset.total_df.names.isin(sample_name)].stats150
    stats_df_150 = pd.DataFrame.from_records(list(train_df_150))
    
    train_df_300 = train_dataset.dataset.total_df[train_dataset.dataset.total_df.names.isin(sample_name)].stats300
    stats_df_300 = pd.DataFrame.from_records(list(train_df_300))
    
    return torch.tensor(list(stats_df_150.mean())),torch.tensor(list(stats_df_150.std())),torch.tensor(list(stats_df_300.mean())),torch.tensor(list(stats_df_300.std()))

def add_transforms(transform,meanBF_150,stdBF_150,meanBF_300,stdBF_300 ):
        
    if 'normalize' in transform:
        transform.update({'normalize':[transform['normalize'][0],transform['normalize'][1],meanBF_150,stdBF_150,meanBF_300,stdBF_300 ]})  
    if 'scale' in transform:
        transform.update({'scale':[transform['scale'][0],meanBF_150,stdBF_150,meanBF_300,stdBF_300 ]})  
    return transform

def val_test_indices(dataset,minor,val_test_size):
    minor.sort()
    TestValSamples = [dataset.samples[i] for i in minor]
    TestValTargets = [dataset.targets[i] for i in minor]
    
    # split the samples in a validation and a test set
    test_indices_list,val_selected_indices = dataset_split(TestValSamples, TestValTargets, val_test_size[1]/(val_test_size[0]+val_test_size[1]))
    Test_indices = [minor[i] for i in test_indices_list]
    Val_indices = [minor[i] for i in val_selected_indices]
    return Val_indices,Test_indices


def Dataloader_train_val_test(path,pathBF,extension,val_test_size,transform,batch_size,num_workers,mode):
    
    # load the BF data
    total_np = np.load(pathBF,allow_pickle=True)
    total_df = pd.DataFrame(total_np, columns=['names','zones','stats150','category','stats300'])     
      
    
    # make the initial dataset
    if extension == ('.jpg'):
        fulset = Fullset_vhr(path,total_df,extension)
    if extension == ('.npy'):
        fulset = Fullset_hr(path,total_df,extension)
    # initial training testing/val split
    main,minor = dataset_split(fulset.samples, fulset.targets,1-(val_test_size[0]+val_test_size[1])) 
    

    # get the split between validation and testing
    Val_indices,Test_indices = val_test_indices(fulset,minor,val_test_size)
    
    # if mode is test add the val indices to main
    if mode=='test':
        #extend the indices of train with the indices of val
        main.extend(Val_indices)    
    
    # mean and std of BF dataset using only training samples
    meanBF_150,stdBF_150,meanBF_300,stdBF_300 = BF_stats(fulset,main)
    
    # adding BF stats to transforms
    if transform==[]:
        STATS_dataset = Subset(fulset,main,transform)
        STATS_loader = torch.utils.data.DataLoader(STATS_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=False)
        mean, std = get_mean_and_std(STATS_loader)
        transform = {'augment':'rand_flip','normalize':[mean,std,meanBF_150,stdBF_150,meanBF_300,stdBF_300 ]}
    else:     
        transform = add_transforms(transform,meanBF_150,stdBF_150,meanBF_300,stdBF_300 )
    
    # create the training set with transform
    train_dataset = Subset(fulset,main,transform)
    
    # remove the augmentation for the test and val set
    transform.pop('augment', None)
    test_dataset = Subset(fulset,Test_indices,transform)
    val_dataset = Subset(fulset,Val_indices,transform)

    # create the dataloaders 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size ,num_workers=num_workers, shuffle=False)
    
    return train_loader,val_loader,test_loader


if __name__ == "__main__":        
    path = r''
    pathBF = r""

    val_test_size = 0.2
    batch_size = 10
    num_workers = 0        
    transform = {'augment':'rand_flip','scale':[torch.tensor([[[255]],[[ 255]],[[ 255]]])]}
    transform = []
    extension =  ('.jpg')
    
    total_df = pd.read_pickle(pathBF)    
    
    fulset = Fullset_vhr(path,total_df,('.jpg'))
    TESTING = torch.utils.data.DataLoader(fulset, batch_size=1 ,num_workers=0, shuffle=False)
    test, target = next(iter(TESTING))
    
    main,minor = dataset_split(fulset.samples, fulset.targets,1-val_test_size ) 
    
    meanBF,stdBF = BF_stats(fulset,main)
    
    test, target = next(iter(train_loader))
    
