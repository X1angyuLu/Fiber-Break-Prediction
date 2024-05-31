import re
import os
import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def get_file_list(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    sorted_file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))
    return sorted_file_list

class dataset(DataLoader):
    def __init__(self, file_path1, file_path2, force, start_index = 0, end_index = 720):
        self.file_path1 = file_path1
        self.fname_list1 = get_file_list(file_path1)
        self.fname_list1 = self.fname_list1[start_index:end_index]
        self.file_path2 = file_path2
        self.fname_list2 = get_file_list(file_path2)
        self.fname_list2 = self.fname_list2[start_index:end_index]
        self.force = force


    def __len__(self) -> int:
        return len(self.fname_list1)
    
    def __getitem__(self, item):
        image1 = nib.load(self.file_path1 + self.fname_list1[item])
        image1 = image1.get_fdata()
        image1 = image1/255

        image2 = nib.load(self.file_path2 + self.fname_list2[item])
        image2 = image2.get_fdata()
        image2 = image2/255

        return (image1,image2,self.force)