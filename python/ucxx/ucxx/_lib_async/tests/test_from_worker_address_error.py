# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import os
import re
from unittest.mock import patch

import numpy as np
import pytest
import trio

import ucxx
from ucxx._lib_async.utils_test import compute_timeouts
from ucxx.testing import join_processes, terminate_process

mp = mp.get_context("spawn")


def _test_from_worker_address_error_server(q1, q2, error_type, timeout):
    async def run():
        address = bytearray(ucxx.get_worker_address())

        if error_type == "unreachable":
            ucxx.reset()
            q1.put(address)
        else:
            q1.put(address)

            ep_ready = q2.get()
            assert ep_ready == "ready"

            ucxx.reset()

    async def main():
        async with trio.open_nursery() as nursery:
            ucxx.core._init_with_nursery(nursery=nursery)
            with trio.fail_after(timeout):
                await run()

    try:
        trio.run(main)
    finally:
        ucxx.stop_notifier_thread()


def _test_from_worker_address_error_client(q1, q2, error_type, timeout):
    async def run():
        remote_address = ucxx.get_ucx_address_from_buffer(q1.get())
        if error_type == "unreachable":
            server_closed = q1.get()
            assert server_closed == "Server closed"

        if error_type == "unreachable":
            with pytest.raises(
                ucxx.exceptions.UCXError,
                match="Destination is unreachable|Endpoint timeout",
            ):
                ep = await ucxx.create_endpoint_from_worker_address(remote_address)
                while ep.alive:
                    await trio.sleep(0)
                    if not ucxx.core._get_ctx().progress_mode.startswith("thread"):
                        ucxx.progress()
                ep._ep.raise_on_error()
        else:
            ep = await ucxx.create_endpoint_from_worker_address(remote_address)

            if re.match("timeout.*send", error_type):
                q2.put("ready")

                while ep.alive:
                    await trio.sleep(0)
                    if not ucxx.core._get_ctx().progress_mode.startswith("thread"):
                        ucxx.progress()

                with pytest.raises(
                    (
                        ucxx.exceptions.UCXConnectionResetError,
                        ucxx.exceptions.UCXEndpointTimeoutError,
                    )
                ):
                    with trio.fail_after(1.0):
                        if error_type == "timeout_am_send":
                            await ep.am_send(np.zeros(10))
                        else:
                            await ep.send(np.zeros(10), tag=0, force_tag=True)
            else:
                with pytest.raises(
                    (
                        ucxx.exceptions.UCXConnectionResetError,
                        ucxx.exceptions.UCXEndpointTimeoutError,
                    )
                ):
                    q2.put("ready")

                    while ep.alive:
                        await trio.sleep(0)
                        if not ucxx.core._get_ctx().progress_mode.startswith("thread"):
                            ucxx.progress()

                    with trio.fail_after(3.0):
                        if error_type == "timeout_am_recv":
                            await ep.am_recv()
                        else:
                            msg = np.empty(10)
                            await ep.recv(msg, tag=0, force_tag=True)

    async def main():
        async with trio.open_nursery() as nursery:
            ucxx.core._init_with_nursery(nursery=nursery)
            with trio.fail_after(timeout):
                await run()

    try:
        trio.run(main)
    finally:
        ucxx.stop_notifier_thread()


@pytest.mark.parametrize(
    "error_type",
    [
        "unreachable",
        "timeout_am_send",
        "timeout_am_recv",
        "timeout_send",
        "timeout_recv",
    ],
)
@patch.dict(
    os.environ,
    {
        "UCX_WARN_UNUSED_ENV_VARS": "n",
        # Set low timeouts to ensure tests quickly raise as expected
        "UCX_KEEPALIVE_INTERVAL": "100ms",
        "UCX_UD_TIMEOUT": "100ms",
    },
)
def test_from_worker_address_error(pytestconfig, error_type):
    async_timeout, join_timeout = compute_timeouts(pytestconfig)

    q1 = mp.Queue()
    q2 = mp.Queue()

    server = mp.Process(
        target=_test_from_worker_address_error_server,
        args=(q1, q2, error_type, async_timeout),
    )
    server.start()

    client = mp.Process(
        target=_test_from_worker_address_error_client,
        args=(q1, q2, error_type, async_timeout),
    )
    client.start()

    if error_type == "unreachable":
        server.join()
        q1.put("Server closed")

    join_processes([client, server], timeout=join_timeout)
    terminate_process(server)
    try:
        terminate_process(client)
    except RuntimeError as e:
        if ucxx.get_ucx_version() < (1, 12, 0):
            if all(t in error_type for t in ["timeout", "send"]):
                pytest.xfail(
                    "Requires https://github.com/openucx/ucx/pull/7527 with rc/ud."
                )
            elif all(t in error_type for t in ["timeout", "recv"]):
                pytest.xfail(
                    "Requires https://github.com/openucx/ucx/pull/7531 with rc/ud."
                )
        else:
            raise e
