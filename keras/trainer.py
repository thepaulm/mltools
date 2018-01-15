#!/usr/bin/env python

import traceback as tb
from timeit import default_timer as timer
import pplot.pplot as ppt


class Trainer(object):
    def __init__(self, mmf):
        self.mmf = mmf
        self.exception = None
        self.model = None
        self.history = {}
        self.train_time = 0

    def train(self, g, lr=1e-3, epochs=32, verbose=False, optimizer='adam', loss='mean_squared_error'):
        if self.model is None:
            self.model = self.mmf(g)
            self.model.compile(loss=loss, optimizer=optimizer)
        self.model.optimizer.lr = lr

        # Train it
        try:
            tg = g.train_gen()
            vg = g.val_gen()
            start = timer()
            history = self.model.fit_generator(tg, steps_per_epoch=len(tg), epochs=epochs, verbose=verbose,
                                               validation_data=vg, validation_steps=len(vg))
            end = timer()
            self.train_time += end - start
            for k in history.history.keys():
                if k in self.history:
                    self.history[k].extend(history.history[k])
                else:
                    self.history[k] = history.history[k]

        except Exception as e:
            tb.print_exception(type(e), e, tb=None)
            self.exception = e

    def losses(self, offset=0):
        return [self.history[k][offset:] for k in ['loss', 'val_loss']]

    def plot_predict(self, g, index=-1, descale=False):
        if g.has_val():
            dg = g.val_gen()
        else:
            dg = g.train_gen()
        x, y = dg[len(dg)-1]
        p = self.model.predict(x[-1].reshape((1, ) + self.model.layers[0].input_shape[1:]))
        x = x[index]
        y = y[index]
        p = p[-1]
        if descale:
            x = dg.descale(x)
            y = dg.descale(y)
            p = dg.descale(p)
        ppt.pts(x, y, p)

    def plot_losses(self):
        ppt.ptt(*self.losses())
