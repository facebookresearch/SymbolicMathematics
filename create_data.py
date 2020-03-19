# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import os
import sys
import math


if __name__ == '__main__':

    assert len(sys.argv) == 3

    data_path = sys.argv[1]
    trn_path = sys.argv[1] + '.train'
    vld_path = sys.argv[1] + '.valid'
    tst_path = sys.argv[1] + '.test'
    vld_tst_size = int(sys.argv[2])
    assert not os.path.isfile(trn_path)
    assert not os.path.isfile(vld_path)
    assert not os.path.isfile(tst_path)
    assert vld_tst_size > 0

    print(f"Reading data from {data_path} ...")
    with io.open(data_path, mode='r', encoding='utf-8') as f:
        lines = [line for line in f]
    total_size = len(lines)
    print(f"Read {total_size} lines.")
    assert 2 * vld_tst_size < total_size

    alpha = math.log(total_size - 0.5) / math.log(2 * vld_tst_size)
    assert int((2 * vld_tst_size) ** alpha) == total_size - 1
    vld_tst_indices = [int(i ** alpha) for i in range(1, 2 * vld_tst_size + 1)]
    vld_indices = set(vld_tst_indices[::2])
    tst_indices = set(vld_tst_indices[1::2])
    assert len(vld_tst_indices) == 2 * vld_tst_size
    assert max(vld_tst_indices) == total_size - 1
    assert len(vld_indices) == vld_tst_size
    assert len(tst_indices) == vld_tst_size

    # sanity check
    total = 0
    power = 0
    while True:
        a = 10 ** power
        b = 10 * a
        s = len([True for x in vld_tst_indices if a <= x < b and x <= total_size])
        if s == 0:
            break
        print("[%12i %12i[: %i" % (a, b, s))
        total += s
        power += 1
    assert total == 2 * vld_tst_size

    print(f"Writing train data to {trn_path} ...")
    print(f"Writing valid data to {vld_path} ...")
    print(f"Writing test data to {tst_path} ...")
    f_train = io.open(trn_path, mode='w', encoding='utf-8')
    f_valid = io.open(vld_path, mode='w', encoding='utf-8')
    f_test = io.open(tst_path, mode='w', encoding='utf-8')

    for i, line in enumerate(lines):
        if i in vld_indices:
            f_valid.write(line)
        elif i in tst_indices:
            f_test.write(line)
        else:
            f_train.write(line)
        if i % 1000000 == 0:
            print(i, end='...', flush=True)

    f_train.close()
    f_valid.close()
    f_test.close()
