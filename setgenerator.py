#!/usr/bin/env python

import keras
import pandas as pd
import math


class SetGenerator(object):
    def __init__(self, datax, datay, batch_size=32, val_pct=None):
        if isinstance(datax, pd.Series):
            datax = datax.as_matrix()
        if isinstance(datay, pd.Series):
            datay = datay.as_matrix()
        self.datax = datax
        self.datay = datay
        self.batch_size = batch_size
        self.val_pct = val_pct
        if val_pct is not None:
            split = int(len(self.datax) * ((100 - val_pct) / 100.0))
            self.tdx = self.datax[:split]
            self.vdx = self.datax[split:]
            self.tdy = self.datay[:split]
            self.vdy = self.datay[split:]
        else:
            self.tdx = self.datax
            self.tdy = self.datay

    def train_gen(self):
        return SetBatchGenerator(self.tdx, self.tdy, self.batch_size)

    def val_gen(self):
        return SetBatchGenerator(self.vdx, self.vdy, self.batch_size)

    def has_val(self):
        return self.val_pct is not None


class SetBatchGenerator(keras.utils.Sequence):
    def __init__(self, datax, datay, batch_size):
        self.datax = datax
        self.datay = datay
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.datax) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(len(self.datax), start + self.batch_size)
        return self.datax[start:end], self.datay[start:end]
