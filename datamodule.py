import pathlib
import os
import numpy as np
import multiprocessing

import torch
import lightning.pytorch as pl

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path: str,
                 labels_path: str):
        self.data_path = data_path
        self.labels_path = labels_path

        current_dir = os.getcwd()
        data_dir = pathlib.Path(os.path.join(current_dir, data_path))
        self.data_file_names = [str(x.name) for x in data_dir.iterdir() if x.is_file()]


    def __len__(self):
        return len(self.data_file_names)

    def __getitem__(self, 
                    idx: int):

        data_file_name = self.data_file_names[idx]
        current_dir = os.getcwd()

        data_file_path = os.path.join(current_dir, self.data_path, data_file_name)
        label_file_path = os.path.join(current_dir, self.labels_path, data_file_name)

        data = np.load(data_file_path)
        data = torch.tensor(data, dtype=torch.float32).transpose(1, 0)

        labels = np.load(label_file_path)
        labels = torch.tensor(labels, dtype=torch.float64).transpose(1, 0)

        return data, labels

class DataModule(pl.LightningModule):

    def __init__(self,
                 data_path: str,
                 labels_path: str,
                 batch_size: int,
                 num_workers: int = 4,
                 val_split: int = 0.2,
                 **kwargs):

        super().__init__(**kwargs)

        self.data_path = data_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.train_dataset = None
        self.val_dataset = None

    def setup(self,
              stage: str = "idk"):

        if not (self.train_dataset and self.val_dataset):
            dataset = Dataset(data_path=self.data_path,
                              labels_path=self.labels_path)

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [1 - self.val_split, self.val_split])
    
    # DataLoaders need everything to be the same size, so we pad each batch to be the size of the largest item in that batch
    def collate(self,
                batch):
        
        data, labels = zip(*batch)

        padded_data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

        return padded_data, padded_labels

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True,
                                           collate_fn=self.collate)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           collate_fn=self.collate)
            
            




