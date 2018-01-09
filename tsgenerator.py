#!/usr/bin/env python

import keras
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TSGenerator:
    def __init__(self, ts, observations, predictions,
                 batch_size=32, scale=True, val_pct=None):
        '''ts is the timeseries data, obervations/predictions is count of each, batch_size
           is batch_size, scale is boolean saying whether to auto-scale features. If
           you pass scale=False, you can still call scale_fit and scale, it just means
           it won't happen automatically from __getitem__'''
        if isinstance(ts, pd.Series):
            ts = ts.as_matrix()
        self.ts = ts
        self.batch_size = batch_size
        self.observations = observations
        self.predictions = predictions
        self.scaler = None
        self.auto_scale = scale
        self.val_pct = val_pct
        if self.auto_scale is True:
            self.scaler = MinMaxScaler()
        if len(self.ts) < (self.predictions + self.observations):
            raise Exception("Not enough data for samples")
        if val_pct is not None:
            split = int((len(self.ts) - (self.predictions + self.observations)) *
                        (val_pct / 100.0))
            self.tts = ts[:split + self.predictions]
            self.vts = ts[split:]
        else:
            self.tts = ts

    def train_gen(self):
        return BatchGenerator(self.tts, self.observations, self.predictions,
                              self.batch_size, self.auto_scale)

    def val_gen(self):
        return BatchGenerator(self.vts, self.observations, self.predictions,
                              self.batch_size, self.auto_scale)


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, ts, observations, predictions, batch_size, auto_scale):
        self.batch_size = batch_size
        self.ts = ts
        self.observations = observations
        self.predictions = predictions
        self.idx = 0
        self.scaler = None
        self.auto_scale = auto_scale
        if self.auto_scale is True:
            self.scaler = MinMaxScaler()
        if len(self.ts) < (self.predictions + self.observations):
            raise Exception("Not enough data for samples")

    def __len__(self):
        '''return len of batched samples'''
        return math.ceil((len(self.ts) - (self.predictions + self.observations)) / self.batch_size)

    def __getitem__(self, idx):
        '''indexed iterator from sample data'''
        outX = []
        outY = []
        for i in range(self.idx,
                       min(self.idx + self.batch_size,
                           len(self.ts) - (self.observations + self.predictions))):
            d = self.ts[i:i + self.observations + self.predictions]

            x = d[:self.observations]
            y = d[self.observations:]

            if self.auto_scale is not False:
                x = x.reshape(x.shape + (1,)).astype('float64')
                y = y.reshape(y.shape + (1,)).astype('float64')
                self.scaler.fit(x)
                x = np.array(self.scaler.transform(x)).squeeze()
                y = np.array(self.scaler.transform(y)).squeeze()

            outX.append(x)
            outY.append(y)

        x = np.array(outX)
        y = np.array(outY)

        return x.reshape(x.shape + (1, )), y

    def _scale_shape(self, data):
        orig_shape = None
        if len(data.shape) == 1:
            data = data.reshape(data.shape + (1, ))
        if len(data.shape) == 3:
            orig_shape = data.shape
            data = data.squeeze()
        return data, orig_shape

    def _scale_deshape(self, data, orig_shape):
        if orig_shape is not None:
            data = data.reshape(orig_shape)
        return data

    def scale_fit(self, data):
        '''fit scaler to this data (not necessary if scale=True)'''
        data, orig_shape = self._scale_shape(data)
        if self.scaler is None:
            self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(data.astype('float64'))
        return self._scale_deshape(data, orig_shape)

    def scale(self, data):
        '''scale this data from the fitted scaler (not necessary if scale=True)'''
        if self.scaler is None:
            return data

        data, orig_shape = self._scale_shape(data)
        data = self.scaler.transform(data.astype('float64'))
        data = self._scale_deshape(data, orig_shape)
        return data

    def descale(self, data):
        '''descale the data from scaler. Will be noop if no scaler'''
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data
