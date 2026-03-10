# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import threading
import warnings
import weakref
from queue import Queue

import trio

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.exceptions import UCXMessageTruncatedError
from ucxx.types import Tag

from .continuous_ucx_progress import BlockingMode, PollingMode, ThreadMode
from .endpoint import Endpoint
from .exchange_peer_info import exchange_peer_info
from .listener import ActiveClients, Listener, _listener_handler
from .notifier_thread import _notifierThread
from .utils import hash64bits

logger = logging.getLogger("ucx")


_progress_started = False


def _reset_progress_flag():
    global _progress_started
    _progress_started = False


class ApplicationContext:
    """
    The context of the trio interface of UCX.

    Parameters
    ----------
    config_dict : dict
        UCX configuration options.
    nursery : trio.Nursery
        A long-lived trio nursery used for spawning internal background
        tasks (progress loop, connection handlers, etc.).  The nursery
        must remain open for the lifetime of this context.
    progress_mode : str or None
        One of ``"thread"``, ``"thread-polling"``, ``"polling"``,
        ``"blocking"``.  Defaults to ``"thread"``.
    enable_delayed_submission : bool or None
        Enable delayed request submission (requires thread progress).
    connect_timeout : float or None
        Timeout in seconds for endpoint peer-info exchange.
    """

    _progress_mode = None
    _enable_delayed_submission = None

    def __init__(
        self,
        config_dict={},
        nursery=None,
        progress_mode=None,
        enable_delayed_submission=None,
        connect_timeout=None,
    ):
        if nursery is None:
            raise TypeError(
                "ApplicationContext requires a trio nursery.  Pass "
                "nursery=<trio.Nursery> so that background tasks can be "
                "spawned."
            )
        self._nursery = nursery
        self._trio_token = trio.lowlevel.current_trio_token()
        self.notifier_thread_q = None
        self.notifier_thread = None
        self._listener_active_clients = ActiveClients()
        self._next_listener_id = 0

        self.progress_mode = progress_mode
        self.enable_delayed_submission = enable_delayed_submission

        if connect_timeout is None:
            self.connect_timeout = float(os.environ.get("UCXPY_CONNECT_TIMEOUT", 5))
        else:
            self.connect_timeout = connect_timeout

        self.context = ucx_api.UCXContext(config_dict)
        self.worker = ucx_api.UCXWorker(
            self.context,
            enable_delayed_submission=self.enable_delayed_submission,
            enable_python_future=False,
        )

        self.start_notifier_thread()

        weakref.finalize(self, _reset_progress_flag)

        # Ensure progress even before Endpoints get created.
        self.continuous_ucx_progress()

    @property
    def progress_mode(self):
        return self._progress_mode

    @progress_mode.setter
    def progress_mode(self, progress_mode):
        if self._progress_mode is None:
            if progress_mode is None:
                if "UCXPY_PROGRESS_MODE" in os.environ:
                    progress_mode = os.environ["UCXPY_PROGRESS_MODE"]
                else:
                    progress_mode = "thread"

            valid_progress_modes = ["blocking", "polling", "thread", "thread-polling"]
            if not isinstance(progress_mode, str) or not any(
                progress_mode == m for m in valid_progress_modes
            ):
                raise ValueError(
                    f"Unknown progress mode '{progress_mode}', valid modes are: "
                    "'blocking', 'polling', 'thread' or 'thread-polling'"
                )

            self._progress_mode = progress_mode
        else:
            raise RuntimeError("Progress mode already set, modifying not allowed")

    @property
    def enable_delayed_submission(self):
        return self._enable_delayed_submission

    @enable_delayed_submission.setter
    def enable_delayed_submission(self, enable_delayed_submission):
        if self._enable_delayed_submission is None:
            if enable_delayed_submission is None:
                if "UCXPY_ENABLE_DELAYED_SUBMISSION" in os.environ:
                    explicit_enable_delayed_submission = (
                        False
                        if os.environ["UCXPY_ENABLE_DELAYED_SUBMISSION"] == "0"
                        else True
                    )
                else:
                    explicit_enable_delayed_submission = self.progress_mode.startswith(
                        "thread"
                    )
            else:
                explicit_enable_delayed_submission = enable_delayed_submission

            if (
                not self.progress_mode.startswith("thread")
                and explicit_enable_delayed_submission
            ):
                raise ValueError(
                    f"Delayed submission requested, but '{self.progress_mode}' does "
                    "not support it, 'thread' or 'thread-polling' progress mode "
                    "required."
                )

            self._enable_delayed_submission = explicit_enable_delayed_submission
        else:
            raise RuntimeError(
                "Enable delayed submission already set, modifying not allowed"
            )

    @property
    def config(self):
        """UCX configuration options as a dict."""
        return self.context.config

    @property
    def ucp_context_info(self):
        """Low-level UCX info about this endpoint as a string."""
        return self.context.info

    @property
    def ucp_worker(self):
        """The underlying UCP worker handle (ucp_worker_h) as a Python integer."""
        return self.worker.handle

    @property
    def ucxx_worker(self):
        """The underlying UCXX worker pointer (ucxx::Worker*) as a Python integer."""
        return self.worker.ucxx_ptr

    @property
    def ucp_worker_info(self):
        """Return low-level UCX info about this endpoint as a string."""
        return self.worker.info

    @property
    def worker_address(self):
        return self.worker.address

    def start_notifier_thread(self):
        if self.worker.enable_python_future and self.notifier_thread is None:
            logger.debug("UCXX_ENABLE_PYTHON available, enabling notifier thread")
            self.notifier_thread_q = Queue()
            self.notifier_thread = threading.Thread(
                target=_notifierThread,
                args=(self._trio_token, self.worker, self.notifier_thread_q),
                name="UCX-Py Async Notifier Thread",
            )
            self.notifier_thread.start()
        else:
            logger.debug(
                "Python future disabled under trio, notifier thread not needed"
            )

    def stop_notifier_thread(self):
        """
        Stop Python future notifier thread.

        .. warning:: When the notifier thread is enabled it may be necessary to
                     explicitly call this method before shutting down the process,
                     otherwise it may block indefinitely waiting for the thread to
                     terminate. Executing ``ucxx.reset()`` will also run this method.
        """
        if self.notifier_thread_q and self.notifier_thread:
            self.notifier_thread_q.put("shutdown")
            while True:
                self.notifier_thread.join(timeout=0.01)
                if not self.notifier_thread.is_alive():
                    self.notifier_thread = None
                    break
            logger.debug("Notifier thread stopped")
        else:
            logger.debug("Notifier thread not running")

    def create_listener(
        self,
        callback_func,
        port=0,
        endpoint_error_handling=True,
        connect_timeout=5.0,
    ):
        """Create and start a listener to accept incoming connections

        callback_func is the function or coroutine that takes one
        argument -- the Endpoint connected to the client.

        Notice, the listening is closed when the returned Listener
        goes out of scope thus remember to keep a reference to the object.

        Parameters
        ----------
        callback_func: function or coroutine
            A callback function that gets invoked when an incoming
            connection is accepted
        port: int, optional
            An unused port number for listening, or `0` to let UCX assign
            an unused port.
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.
        connect_timeout: float
            Timeout in seconds for exchanging peer info.

        Returns
        -------
        Listener
            The new listener. When this object is deleted, the listening stops
        """
        self.continuous_ucx_progress()
        if port is None:
            port = 0

        logger.info("create_listener() - Start listening on port %d" % port)
        listener_id = self._next_listener_id
        self._next_listener_id += 1
        ret = Listener(
            ucx_api.UCXListener.create(
                worker=self.worker,
                port=port,
                cb_func=_listener_handler,
                cb_args=(
                    self._trio_token,
                    self._nursery,
                    callback_func,
                    self,
                    endpoint_error_handling,
                    connect_timeout,
                    listener_id,
                    self._listener_active_clients,
                ),
                deliver_endpoint=True,
            ),
            listener_id,
            self._listener_active_clients,
        )
        return ret

    async def create_endpoint(
        self,
        ip_address,
        port,
        endpoint_error_handling=True,
        connect_timeout=5.0,
    ):
        """Create a new endpoint to a server

        Parameters
        ----------
        ip_address: str
            IP address of the server the endpoint should connect to
        port: int
            IP address of the server the endpoint should connect to
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled.
        connect_timeout: float
            Timeout in seconds for exchanging peer info.

        Returns
        -------
        Endpoint
            The new endpoint
        """
        self.continuous_ucx_progress()

        ucx_ep = ucx_api.UCXEndpoint.create(
            self.worker, ip_address, port, endpoint_error_handling
        )
        if not self.progress_mode.startswith("thread"):
            self.worker.progress()

        seed = os.urandom(16)
        msg_tag = hash64bits("msg_tag", seed, ucx_ep.handle)
        try:
            peer_info = await exchange_peer_info(
                endpoint=ucx_ep,
                msg_tag=msg_tag,
                listener=False,
                connect_timeout=connect_timeout,
            )
        except UCXMessageTruncatedError as e:
            ucx_ep.raise_on_error()
            raise e

        tags = {
            "msg_send": peer_info["msg_tag"],
            "msg_recv": msg_tag,
        }
        ep = Endpoint(endpoint=ucx_ep, ctx=self, tags=tags)

        logger.debug(
            "create_endpoint() client: %s, error handling: %s, msg-tag-send: %s, "
            "msg-tag-recv: %s"
            % (
                hex(ep._ep.handle),
                endpoint_error_handling,
                hex(ep._tags["msg_send"]),
                hex(ep._tags["msg_recv"]),
            )
        )

        return ep

    async def create_endpoint_from_worker_address(
        self,
        address,
        endpoint_error_handling=True,
    ):
        """Create a new endpoint to a server

        Parameters
        ----------
        address: UCXAddress
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling.

        Returns
        -------
        Endpoint
            The new endpoint
        """
        self.continuous_ucx_progress()

        ucx_ep = ucx_api.UCXEndpoint.create_from_worker_address(
            self.worker,
            address,
            endpoint_error_handling,
        )
        if not self.progress_mode.startswith("thread"):
            self.worker.progress()

        ep = Endpoint(endpoint=ucx_ep, ctx=self, tags=None)

        logger.debug(
            "create_endpoint() client: %s, error handling: %s"
            % (hex(ep._ep.handle), endpoint_error_handling)
        )

        return ep

    def continuous_ucx_progress(self):
        """Guarantees continuous UCX progress.

        This is called automatically when calling ``create_listener()``
        or ``create_endpoint()``.  Background tasks are spawned into the
        nursery that was passed to the constructor.
        """
        global _progress_started
        if _progress_started:
            return

        logger.info(f"Starting progress in '{self.progress_mode}' mode")

        if self.progress_mode == "thread":
            self._progress_task = ThreadMode(
                self.worker, self._nursery, polling_mode=False
            )
        elif self.progress_mode == "thread-polling":
            self._progress_task = ThreadMode(
                self.worker, self._nursery, polling_mode=True
            )
        elif self.progress_mode == "polling":
            self._progress_task = PollingMode(self.worker, self._nursery)
        elif self.progress_mode == "blocking":
            self._progress_task = BlockingMode(self.worker, self._nursery)

        _progress_started = True

    def get_ucp_worker(self):
        """Returns the underlying UCP worker handle (ucp_worker_h)
        as a Python integer.
        """
        warnings.warn(
            "ApplicationContext.get_ucp_worker() is deprecated and will soon "
            "be removed, use the ApplicationContext.ucp_worker property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.ucp_worker

    def get_ucxx_worker(self):
        """Returns the underlying UCXX worker pointer (ucxx::Worker*)
        as a Python integer.
        """
        warnings.warn(
            "ApplicationContext.get_ucxx_worker() is deprecated and will soon "
            "be removed, use the ApplicationContext.ucxx_worker property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.ucxx_worker

    def get_config(self):
        """Returns all UCX configuration options as a dict.

        Returns
        -------
        dict
            The current UCX configuration options
        """
        warnings.warn(
            "ApplicationContext.get_config() is deprecated and will soon "
            "be removed, use the ApplicationContext.config property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.config

    def get_worker_address(self):
        warnings.warn(
            "ApplicationContext.get_worker_address() is deprecated and will soon "
            "be removed, use the ApplicationContext.worker_address property instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.worker.address

    def tag_probe(self, tag, remove=False):
        """Probe for tag messages directly on worker without a local Endpoint.

        Parameters
        ----------
        tag: hashable
            Set a tag that must match the received message.
        remove: bool
            If true, remove the message from the queue.

        Returns
        -------
        TagProbeResult
        """
        if not isinstance(tag, Tag):
            tag = Tag(tag)

        return self.worker.tag_probe(tag, remove=remove)

    async def recv(self, buffer, tag):
        """Receive directly on worker without a local Endpoint into `buffer`.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into.
        tag: hashable, optional
            Set a tag that must match the received message.
        """
        if not isinstance(buffer, Array):
            buffer = Array(buffer)
        if not isinstance(tag, Tag):
            tag = Tag(tag)
        nbytes = buffer.nbytes
        log = "[Worker Recv] worker: %s, tag: %s, nbytes: %d, type: %s" % (
            hex(self.worker.handle),
            hex(tag.value),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)

        req = self.worker.tag_recv(buffer, tag)
        return await req.wait()

    async def recv_with_handle(self, buffer, probe_result):
        """Receive tag message using message handle obtained from tag_probe.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into.
        probe_result: TagProbeResult
            The probe result obtained from tag_probe with remove=True.

        Returns
        -------
        The received data
        """
        if not isinstance(buffer, Array):
            buffer = Array(buffer)

        nbytes = buffer.nbytes
        log = "[Worker Recv With Handle] worker: %s, nbytes: %d, type: %s" % (
            hex(self.worker.handle),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)

        req = self.worker.tag_recv_with_handle(buffer, probe_result)
        return await req.wait()
