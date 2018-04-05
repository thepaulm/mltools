#!/usr/bin/env python

from __future__ import print_function

import os
import subprocess
import pandas as pd
import numpy as np
import keras
import math


class FileInfo(object):
    def __init__(self, filename, index, startrow, endrow, total_start_row):
        self.filename = filename
        self.index = index
        self.startrow = startrow
        self.endrow = endrow
        self.total_start_row = total_start_row


class FileLoc(object):
    def __init__(self, finfo, offset):
        self.finfo = finfo
        self.offset = offset


class CSVFileGenerator(object):
    def __init__(self, directory, ycolumns, xcolumns=None, batch_size=32,
                 val_pct=None, shuffle=True, handler='pandas', skip_header=0):
        '''Directory is where to look for the files.
           ycolumns is the label columns.
           xcolumns is the feature columns (default None for "all the non-labels").
           batch_size is batch size per epoch.
           val_pct is percentage (0-100) for validation data - can be 0.
           shuffle default True for shuffle each batch.
           handler="pandas" for pd dataframe handling. This is mostly useful for
             variable data types and column names. This may also be "numpy" which
             will use less memory (and is faster) but won't parse column headers.
             In the case of handler="numpy", ycolumns and xcolumns must be index
             ranges instead of label strings.
        '''

        self.directory = directory
        if handler == "pandas":
            if isinstance(ycolumns, str):
                ycolumns = [ycolumns]
            if isinstance(xcolumns, str):
                xcolumns = [xcolumns]
        elif handler == "numpy":
            if isinstance(ycolumns, int):
                ycolumns = slice(ycolumns, ycolumns+1)
            if isinstance(xcolumns, int):
                xcolumns = slice(xcolumns, xcolumns+1)
        else:
            raise Exception("no handler " + handler)

        self.handler = handler
        self.skip_header = skip_header
        self.ycolumns = ycolumns
        self.xcolumns = xcolumns
        if val_pct == 0:
            val_pct = None
        self.val_pct = val_pct
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.files = []
        self.total_lines = 0

        # map files files and line counts
        self.build_file_map()
        self.train_line_count = self.total_lines
        self.val_line_count = None

        self.start_loc = FileLoc(self.files[0], 0)
        self.end_loc = FileLoc(self.files[-1], self.files[-1].endrow+1)
        self.split_loc = self.end_loc

        # figure out where to do the val split
        if self.val_pct:
            split = int(self.total_lines * ((100.0 - val_pct) / 100.0))
            self.train_line_count = split
            self.val_line_count = self.total_lines - split
            lines = 0
            for i, f in enumerate(self.files):
                if lines + f.endrow + 1 > split:
                    offset = split - lines
                    self.split_loc = FileLoc(f, offset)
                    break
                lines += f.endrow + 1

    def get_file_lines(self, fullpath):
        p = subprocess.Popen('wc -l ' + fullpath, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return int(out.split()[0])

    def _sorted_listdir(self, directory):
        def pmtime(f):
            return os.path.getmtime(directory + '/' + f)
        return sorted(os.listdir(directory), key=pmtime, reverse=False)

    def build_file_map(self):
        files = [f for f in self._sorted_listdir(self.directory) if f.endswith('.csv')]
        for i, f in enumerate(files):
            fullpath = self.directory + '/' + f
            lines = self.get_file_lines(fullpath)
            if self.handler == "pandas":
                if lines > 0:
                    lines -= 1  # Remove the header
            elif self.skip_header != 0:
                if lines > self.skip_header:
                    lines -= self.skip_header
            if lines > 0:
                self.files.append(FileInfo(fullpath, i, 0, lines-1, self.total_lines))
                self.total_lines += lines

    def has_val(self):
        return self.val_pct is not None

    def train_gen(self):
        return CSVFileBatchGenerator(self.directory, self.ycolumns, self.xcolumns,
                                     self.handler, self.skip_header, self.batch_size,
                                     self.train_line_count, self.files, self.shuffle,
                                     start_loc=self.start_loc, end_loc=self.split_loc)

    def val_gen(self):
        if not self.has_val():
            return None
        return CSVFileBatchGenerator(self.directory, self.ycolumns, self.xcolumns,
                                     self.handler, self.skip_header, self.batch_size,
                                     self.val_line_count, self.files, self.shuffle,
                                     start_loc=self.split_loc, end_loc=self.end_loc)


class CSVFileBatchGenerator(keras.utils.Sequence):
    def __init__(self, directory, ycolumns, xcolumns, handler, skip_header,
                 batch_size, line_count, files, shuffle, start_loc, end_loc):
        self.directory = directory
        self.ycolumns = ycolumns
        self.xcolumns = xcolumns
        self.batch_size = batch_size
        self.line_count = line_count
        self.files = files
        self.shuffle = shuffle
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.current_file_index = self.start_loc.finfo.index
        self.total_offset_row = start_loc.finfo.total_start_row + start_loc.offset

        if handler == "pandas":
            self.read_cur = self.read_curdf
            self.append_data = self.append_df
            self.shuffler = self.shuffle_df
            self.make_x = self.make_x_df
            self.make_y = self.make_y_df
            if skip_header != 0:
                raise Exception("pandas doesn't skip header")
        elif handler == "numpy":
            self.read_cur = self.read_curnp
            self.append_data = self.append_np
            self.shuffler = self.shuffle_np
            self.skip_header = skip_header
            self.make_x = self.make_x_np
            self.make_y = self.make_y_np
        else:
            raise Exception("Uknown handler " + handler)
        self.handler = handler  # Nobody uses this, but this for debugging

        self.read_cur(self.start_loc.finfo)

    # Handler Specifics
    def read_curdf(self, finfo):
        self.curdata = pd.read_csv(finfo.filename)

    def read_curnp(self, finfo):
        self.curdata = np.loadtxt(finfo.filename, delimiter=',', skiprows=self.skip_header)

    def append_df(self, df1, df2):
        return df1.append(df2)

    def append_np(self, np1, np2):
        return np.concatenate((np1, np2))

    def shuffle_df(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    def shuffle_np(self, np1):
        return np.random.shuffle(np1)

    def make_x_df(self, df):
        if self.xcolumns is not None:
            return df[self.xcolumns]
        return df[[d for d in df.columns if d not in self.ycolumns]].as_matrix()

    def make_y_df(self, df):
        return df[self.ycolumns].as_matrix()

    def make_x_np(self, np1):
        return np1[:, self.xcolumns]

    def make_y_np(self, np1):
        return np1[:, self.ycolumns]

    # Generic Methods
    def __len__(self):
        return math.ceil(self.line_count / self.batch_size)

    def _set_current_file(self, i):
            self.current_file_index = i
            current = self.files[self.current_file_index]
            self.read_cur(current)
            return current

    def __getitem__(self, idx):
        start = idx * self.batch_size + self.total_offset_row
        count = self.batch_size
        current = self.files[self.current_file_index]

        # Get to the right file
        # If we're before, start at 0 ...
        if start < current.total_start_row:
            current = self._set_current_file(0)

        # Move forward until we are at the right one ...
        if start > current.total_start_row + current.endrow:
            while start > current.total_start_row + current.endrow:
                self.current_file_index += 1
                current = self.files[self.current_file_index]
            self.read_cur(current)

        start = start - current.total_start_row
        copied = 0

        def fix_end(count, copied, start):
            end = len(self.curdata)
            if self.end_loc.finfo.index == self.current_file_index:
                end = self.end_loc.offset
            end = min(count - copied + start, end)
            return end

        end = fix_end(count, copied, start)

        df = self.curdata[start:end]
        copied += end - start

        while copied < count:
            if self.current_file_index + 1 > self.end_loc.finfo.index:
                break
            current = self._set_current_file(self.current_file_index + 1)

            end = fix_end(count, copied, 0)

            df = self.append_data(df, self.curdata[0:end])
            copied += end

        if self.shuffle:
            df = self.shuffler(df)

        return self.make_x(df), self.make_y(df)


#
# Unit tests below here ...
#

def test_dir():
    import tempfile
    td = tempfile.gettempdir() + '/csvfilegeneratortest/'
    return td


def make_test_files(files, lines):
    td = test_dir()
    if os.path.isdir(td):
        clean_test_files(td)
    os.mkdir(td)

    tot_lines = 0
    for f in range(files):
        with open(td + str(f) + '.csv', 'w') as f:
            print("x,y", file=f)
            for i in range(lines):
                print("%d,%d" % (i, tot_lines), file=f)
                tot_lines += 1

    return td


def clean_test_files(td):
    import shutil
    shutil.rmtree(td)


def test_gen(files, lines, batch_size, val_pct, csvgen):

    tg = csvgen.train_gen()
    vg = csvgen.val_gen()

    if val_pct is not None:
        split = int(files * lines * ((100.0 - val_pct) / 100.0))
    else:
        split = files * lines

    checked = 0
    for i in range(len(tg)):
        x, y = tg[i]
        if i < len(tg) - 1:
            assert(len(x) == batch_size)
            assert(len(y) == batch_size)
        assert(len(x) == len(y))
        for j in range(len(x)):
            assert(x[j] == checked % lines)
            assert(y[j] == checked)
            checked += 1

    assert(checked == split)
    train_count = checked

    if val_pct is not None:
        for i in range(len(vg)):
            x, y = vg[i]
            if i < len(vg) - 1:
                assert(len(x) == batch_size)
                assert(len(y) == batch_size)
            assert(len(x) == len(y))
            for j in range(len(x)):
                assert(x[j] == checked % lines)
                assert(y[j] == checked)
                checked += 1
        val_count = checked - train_count
        assert(val_count + train_count == files * lines)


def test_gens(files, lines, batch_size, val_pct, td):

    csvgen = CSVFileGenerator(directory=td, ycolumns='y', batch_size=batch_size,
                              val_pct=val_pct, shuffle=False)
    test_gen(files, lines, batch_size, val_pct, csvgen)

    csvgen = CSVFileGenerator(directory=td, ycolumns=1, xcolumns=0, batch_size=batch_size,
                              val_pct=val_pct, shuffle=False, handler='numpy', skip_header=1)
    test_gen(files, lines, batch_size, val_pct, csvgen)


def test():
    def breaker(type, value, tb):
        import traceback
        import ipdb
        traceback.print_exception(type, value, tb)
        print()
        ipdb.pm()

    import sys
    sys.excepthook = breaker

    files = 20
    lines = 100
    batch_size = 32
    val_pcts = [14, 15, None]
    td = make_test_files(files, lines)

    for val_pct in val_pcts:
        test_gens(files, lines, batch_size, val_pct, td)

    clean_test_files(td)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()

    if args.test:
        test()
