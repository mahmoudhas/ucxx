# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import time
import weakref

import trio

from ucxx._lib.libucxx import UCXWorker


class ProgressTask:
    """Base for progress-mode strategies.

    Parameters
    ----------
    worker : UCXWorker
        The UCX worker context to progress.
    nursery : trio.Nursery
        Nursery used to spawn long-running progress tasks.
    """

    def __init__(self, worker: UCXWorker, nursery: trio.Nursery):
        self.worker = worker
        self.nursery = nursery


def _create_context():
    try:
        from ucxx._cuda_context import ensure_cuda_context

        ensure_cuda_context(0)
    except ImportError:
        import cupy

        cupy.cuda.Device(0).use()


class ThreadMode(ProgressTask):
    def __init__(self, worker, nursery, polling_mode=False):
        super().__init__(worker, nursery)
        worker.set_progress_thread_start_callback(_create_context)
        worker.start_progress_thread(polling_mode=polling_mode, epoll_timeout=1)
        weakref.finalize(self, worker.stop_progress_thread)


class PollingMode(ProgressTask):
    def __init__(self, worker, nursery):
        super().__init__(worker, nursery)
        self.worker.init_blocking_progress_mode()
        nursery.start_soon(self._progress_task)

    async def _progress_task(self):
        """Maintain a UCX progress loop, yielding to trio between iterations."""
        while True:
            worker = self.worker
            if worker is None:
                return
            worker.progress()
            await trio.lowlevel.checkpoint()


class BlockingMode(ProgressTask):
    def __init__(
        self,
        worker: UCXWorker,
        nursery: trio.Nursery,
        progress_timeout: float = 1.0,
    ):
        """Progress the UCX worker in blocking mode.

        Watches the worker's epoll file descriptor for readability, and
        falls back to periodic progress every *progress_timeout* seconds
        to prevent deadlocks.

        Parameters
        ----------
        worker : UCXWorker
            Worker object from the UCXX Cython API to progress.
        nursery : trio.Nursery
            Nursery in which to spawn the watcher tasks.
        progress_timeout : float
            Maximum seconds between forced progress calls.
        """
        super().__init__(worker, nursery)
        self._progress_timeout = progress_timeout
        self.worker.init_blocking_progress_mode()
        nursery.start_soon(self._epoll_watcher)
        nursery.start_soon(self._periodic_progress)

    async def _epoll_watcher(self):
        """Wait for events on the UCX epoll fd, then progress."""
        epoll_fd = self.worker.epoll_file_descriptor
        while True:
            await trio.lowlevel.wait_readable(epoll_fd)
            if self.worker is None:
                return
            self.worker.progress()

    async def _periodic_progress(self):
        """Safety-net: progress the worker at regular intervals."""
        last = time.monotonic() - self._progress_timeout
        while True:
            worker = self.worker
            if worker is None:
                return
            now = time.monotonic()
            if now - last >= self._progress_timeout:
                last = now
                worker.progress()
            await trio.sleep(self._progress_timeout)
