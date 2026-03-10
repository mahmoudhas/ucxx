# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import numpy as np
import pytest
import trio

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers

msg_sizes = [0] + [2**i for i in range(0, 25, 4)]


def _bytearray_assert_equal(a, b):
    assert a == b


def get_data():
    ret = [
        {
            "allocator": bytearray,
            "generator": lambda n: bytearray(b"m" * n),
            "validator": lambda recv, exp: _bytearray_assert_equal(bytes(recv), exp),
            "memory_type": "host",
        },
        {
            "allocator": partial(np.ones, dtype=np.uint8),
            "generator": partial(np.arange, dtype=np.int64),
            "validator": lambda recv, exp: np.testing.assert_equal(
                recv.view(np.int64), exp
            ),
            "memory_type": "host",
        },
    ]

    try:
        import cupy as cp

        ret.append(
            {
                "allocator": partial(cp.ones, dtype=np.uint8),
                "generator": partial(cp.arange, dtype=np.int64),
                "validator": lambda recv, exp: cp.testing.assert_array_equal(
                    cp.asarray(recv).view(np.int64), exp
                ),
                "memory_type": "cuda",
            }
        )
    except ImportError:
        pass

    return ret


def simple_server(size, recv):
    async def server(ep):
        recv = await ep.am_recv()
        await ep.am_send(recv)
        await ep.close()

    return server


@pytest.mark.trio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("recv_wait", [True, False])
@pytest.mark.parametrize("data", get_data())
async def test_send_recv_am(size, recv_wait, data):
    rndv_thresh = 8192
    ucxx.reset()
    ucxx.core._init_with_nursery(
        options={"RNDV_THRESH": str(rndv_thresh)},
        nursery=ucxx.core._test_nursery,
    )

    msg = data["generator"](size)

    recv = []
    listener = ucxx.create_listener(simple_server(size, recv))
    num_clients = 1
    clients = [
        await ucxx.create_endpoint(ucxx.get_address(), listener.port)
        for i in range(num_clients)
    ]
    if recv_wait:
        await trio.sleep(1)

    async with trio.open_nursery() as send_nursery:
        for c in clients:
            send_nursery.start_soon(c.am_send, msg)

    recv_msgs = [None] * len(clients)

    async def recv_and_store(idx, client):
        recv_msgs[idx] = await client.am_recv()

    async with trio.open_nursery() as recv_nursery:
        for i, c in enumerate(clients):
            recv_nursery.start_soon(recv_and_store, i, c)

    for recv_msg in recv_msgs:
        if data["memory_type"] == "cuda" and msg.nbytes < rndv_thresh:
            np.testing.assert_equal(recv_msg.view(np.int64), msg.get())
        else:
            data["validator"](recv_msg, msg)

    async with trio.open_nursery() as close_nursery:
        for c in clients:
            close_nursery.start_soon(c.close)
    await wait_listener_client_handlers(listener)
