import os
import pandas as pd
import numpy as np
import joblib
import pickle
from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from utils.tools import bcolors
from dataset.dataloader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


class LSTFDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Data = data_dict[cfg.data.data_name]
        self.timeenc = 0 if cfg.data.embed != 'timeF' else 1
        self.freq = cfg.data.freq


    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train = self.Data(
                root_path = self.cfg.data.root_path,
                data_path = self.cfg.data.data_path,
                flag = 'train',
                size = [self.cfg.seq_len, self.cfg.label_len, self.cfg.pred_len],
                features = self.cfg.features,
                target = self.cfg.target,
                timeenc = self.timeenc,
                freq = self.freq,
            )

            self.val = self.Data(
                root_path = self.cfg.data.root_path,
                data_path = self.cfg.data.data_path,
                flag = 'val',
                size = [self.cfg.seq_len, self.cfg.label_len, self.cfg.pred_len],
                features = self.cfg.features,
                target = self.cfg.target,
                timeenc = self.timeenc,
                freq = self.freq,
            )
        elif stage == "test":
            self.test = self.Data(
                root_path = self.cfg.data.root_path,
                data_path = self.cfg.data.data_path,
                flag = 'test',
                size = [self.cfg.seq_len, self.cfg.label_len, self.cfg.pred_len],
                features = self.cfg.features,
                target = self.cfg.target,
                timeenc = self.timeenc,
                freq = self.freq,
            )
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.cfg.optimization.batch_size,
            num_workers=self.cfg.optimization.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.cfg.optimization.batch_size,
            num_workers=self.cfg.optimization.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.cfg.optimization.batch_size,
            num_workers=self.cfg.optimization.num_workers,
            shuffle=False,
            drop_last=True,
        )
