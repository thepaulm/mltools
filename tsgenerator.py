#!/usr/bin/env python

import keras
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TSGenerator(keras.utils.Sequence):
    def __init__(self, ts, observations, predictions, batch_size, scale=True):
        '''ts is the timeseries data, obervations/predictions is count of each, batch_size
           is batch_size, scale is boolean saying whether to auto-scale features. If
           you pass scale=False, you can still call scale_fit and scale, it just means
           it won't happen automatically from __getitem__'''
        self.batch_size = batch_size
        self.ts = ts
        self.observations = observations
        self.predictions = predictions
        self.idx = 0
        self.scaler = None
        self.auto_scale = scale
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
            outX.append(d[:self.observations])
            outY.append(d[self.observations:])
        if self.auto_scale is not False:
            self.scaler.fit(outX)
            return np.array(self.scaler.transform(outX)), np.array(self.scaler.transform(outY))

        return np.array(outX), np.array(outY)

    def scale_fit(self, data):
        '''fit scaler to this data (not necessary if scale=True)'''
        if len(data.shape) == 1:
            data = data.reshape(data.shape + (1, ))
        if self.scaler is None:
            self.scaler = MinMaxScaler()
        return self.scaler.fit_transform(data)

    def scale(self, data):
        '''scale this data from the fitted scaler (not necessary if scale=True)'''
        if self.scaler is not None:
            if len(data.shape) == 1:
                data = data.reshape(data.shape + (1, ))
            return self.scaler.transform(data)
        return data

    def descale(self, data):
        '''descale the data from scaler. Will be noop if no scaler'''
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data
