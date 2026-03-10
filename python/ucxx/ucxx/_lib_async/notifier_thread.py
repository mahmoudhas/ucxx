# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import logging
from concurrent.futures import TimeoutError

import trio

import ucxx._lib.libucxx as ucx_api

logger = logging.getLogger("ucx")


async def _run_request_notifier(worker):
    return worker.run_request_notifier()


def _notifierThread(trio_token, worker, q):
    logger.debug("Starting Notifier Thread")
    shutdown = False

    while True:
        worker.populate_python_futures_pool()
        state = worker.wait_request_notifier(period_ns=int(1e9))  # 1 second timeout

        if not q.empty():
            q_val = q.get()
            if q_val == "shutdown":
                logger.debug("_notifierThread shutting down")
                shutdown = True
            else:
                logger.warning(
                    f"_notifierThread got unknown message from IPC queue: {q_val}"
                )

        if state == ucx_api.PythonRequestNotifierWaitState.Shutdown or shutdown is True:
            break
        elif state == ucx_api.PythonRequestNotifierWaitState.Timeout:
            continue

        try:
            trio.from_thread.run(
                _run_request_notifier, worker, trio_token=trio_token
            )
        except TimeoutError:
            logger.debug("Notifier Thread Result Timeout")
        except Exception as e:
            logger.debug(f"Notifier Thread Result Exception: {e}")

    # Clear all Python futures to ensure no references are held to the
    # `ucxx::Worker` that will prevent destructors from running.
    worker.clear_python_futures_pool()
