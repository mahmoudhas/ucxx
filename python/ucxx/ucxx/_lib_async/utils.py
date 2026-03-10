# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import hashlib
import multiprocessing as mp

mp = mp.get_context("spawn")


def hash64bits(*args):
    """64 bit unsigned hash of `args`"""
    # 64 bits hexdigest
    h = hashlib.sha1(bytes(repr(args), "utf-8")).hexdigest()[:16]
    # Convert to an integer and return
    return int(h, 16)
