# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import trio

import ucxx as ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers


@pytest.mark.trio
async def test_tag_match():
    msg1 = bytes("msg1", "utf-8")
    msg2 = bytes("msg2", "utf-8")

    async def server_node(ep):
        async def send_msg1():
            await ep.send(msg1, tag="msg1")

        async def send_msg2():
            await ep.send(msg2, tag="msg2")

        async with trio.open_nursery() as nursery:
            nursery.start_soon(send_msg1)
            await trio.sleep(1)  # Let msg1 finish
            nursery.start_soon(send_msg2)

    lf = ucxx.create_listener(server_node)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lf.port)
    m1, m2 = (bytearray(len(msg1)), bytearray(len(msg2)))

    recv_m2_done = False

    async def recv_m2():
        nonlocal recv_m2_done
        await ep.recv(m2, tag="msg2")
        recv_m2_done = True

    async with trio.open_nursery() as nursery:
        nursery.start_soon(recv_m2)
        # At this point recv_m2 shouldn't be able to finish because its
        # tag "msg2" doesn't match the servers send tag "msg1"
        await trio.sleep(0.01)
        assert not recv_m2_done
        # "msg1" should be ready
        await ep.recv(m1, tag="msg1")
        assert m1 == msg1

    assert m2 == msg2
    await wait_listener_client_handlers(lf)
