#!/usr/bin/env python

import keras
import math
import numpy as np


class TSGenerator(keras.utils.Sequence):
    def __init__(self, ts, observations, predictions, batch_size):
        self.batch_size = batch_size
        self.ts = ts
        self.observations = observations
        self.predictions = predictions
        self.idx = 0
        if len(self.ts) < (self.predictions + self.observations):
            raise Exception("Not enough data for samples")

    def __len__(self):
        return math.ceil((len(self.ts) - (self.predictions + self.observations)) / self.batch_size)

    def __getitem__(self, idx):
        outX = []
        outY = []
        for i in range(self.idx,
                       min(self.idx + self.batch_size,
                           len(self.ts) - (self.observations + self.predictions))):
            d = self.ts[i:i + self.observations + self.predictions]
            outX.append(d[:self.observations])
            outY.append(d[self.observations:])
        return np.array(outX), np.array(outY)
