# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import gc
import logging
import os
import re
import weakref

from ._lib import libucxx as ucx_api
from ._lib.libucxx import UCXError
from ._lib_async import ApplicationContext

logger = logging.getLogger("ucx")

# The module should only instantiate one instance of the application context
# However, the init of CUDA must happen after all process forks thus we delay
# the instantiation of the application context to the first use of the API.
_ctx = None
_test_nursery = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        if _test_nursery is not None:
            _init_with_nursery(nursery=_test_nursery)
        else:
            raise RuntimeError(
                "UCX is not initialized.  "
                "Use ``async with ucxx.init(): ...``"
            )
    return _ctx


def _init_with_nursery(
    options={},
    env_takes_precedence=False,
    nursery=None,
    progress_mode=None,
    enable_delayed_submission=None,
    connect_timeout=None,
):
    """Low-level init that takes an already-open nursery.

    Prefer the public ``async with ucxx.init(): ...`` API instead.
    """
    global _ctx
    if _ctx is not None:
        raise RuntimeError(
            "UCX is already initiated. Call reset() and init() "
            "in order to re-initate UCX with new options."
        )

    options = options.copy()
    for k, v in options.items():
        env_k = f"UCX_{k}"
        env_v = os.environ.get(env_k)
        if env_v is not None:
            if env_takes_precedence:
                options[k] = env_v
                logger.debug(
                    f"Ignoring option {k}={v}; using environment {env_k}={env_v}"
                )
            else:
                logger.debug(
                    f"Ignoring environment {env_k}={env_v}; using option {k}={v}"
                )

    _ctx = ApplicationContext(
        options,
        nursery=nursery,
        progress_mode=progress_mode,
        enable_delayed_submission=enable_delayed_submission,
        connect_timeout=connect_timeout,
    )


@contextlib.asynccontextmanager
async def init(
    options={},
    env_takes_precedence=False,
    progress_mode=None,
    enable_delayed_submission=None,
    connect_timeout=None,
):
    """Initialise UCX and yield control.

    Use as an async context manager::

        async with ucxx.init():
            listener = ucxx.create_listener(handler, port=0)
            ...

    A trio nursery is created and managed internally for background
    tasks (progress loop, connection handlers).  ``reset()`` is called
    automatically when the block exits.

    Parameters
    ----------
    options : dict, optional
        UCX options sent to the underlying UCX library.
    env_takes_precedence : bool, optional
        Whether environment variables take precedence over *options*.
    progress_mode : str, optional
        ``'thread'`` (default), ``'thread-polling'``, ``'polling'``,
        or ``'blocking'``.
    enable_delayed_submission : bool, optional
        Enable delayed request submission (requires thread progress).
    connect_timeout : float, optional
        Timeout in seconds for endpoint peer-info exchange.
    """
    import trio

    async with trio.open_nursery() as nursery:
        _init_with_nursery(
            options=options,
            env_takes_precedence=env_takes_precedence,
            nursery=nursery,
            progress_mode=progress_mode,
            enable_delayed_submission=enable_delayed_submission,
            connect_timeout=connect_timeout,
        )
        try:
            yield
        finally:
            reset()
            nursery.cancel_scope.cancel()


def reset():
    """Resets the UCX library by shutting down all of UCX.

    The library is initiated at next ``init()`` call.
    """
    stop_notifier_thread()
    global _ctx
    if _ctx is not None:
        weakref_ctx = weakref.ref(_ctx)
        _ctx = None
        gc.collect()
        if weakref_ctx() is not None:
            msg = (
                "Trying to reset UCX but not all Endpoints and/or Listeners "
                "are closed(). The following objects are still referencing "
                "ApplicationContext: "
            )
            for o in gc.get_referrers(weakref_ctx()):
                msg += "\n  %s" % str(o)
            raise UCXError(msg)


def stop_notifier_thread():
    global _ctx
    if _ctx:
        _ctx.stop_notifier_thread()
    else:
        logger.debug("UCX is not initialized.")


def get_ucx_version():
    """Return the version of the underlying UCX installation

    Notice, this function doesn't initialize UCX.

    Returns
    -------
    tuple
        The version as a tuple e.g. (1, 7, 0)
    """
    return ucx_api.get_ucx_version()


def progress():
    """Try to progress the communication layer

    Warning, it is illegal to call this from a call-back function such as
    the call-back function given to create_listener.
    """
    return _get_ctx().worker.progress()


def get_config():
    """Returns all UCX configuration options as a dict.

    If UCX is uninitialized, the options returned are the
    options used if UCX were to be initialized now.
    Notice, this function doesn't initialize UCX.

    Returns
    -------
    dict
        The current UCX configuration options
    """

    if _ctx is None:
        return ucx_api.get_current_options()
    else:
        return _get_ctx().config


def create_listener(
    callback_func,
    port=None,
    endpoint_error_handling=True,
    connect_timeout=5.0,
):
    return _get_ctx().create_listener(
        callback_func,
        port,
        endpoint_error_handling=endpoint_error_handling,
        connect_timeout=connect_timeout,
    )


async def create_endpoint(
    ip_address, port, endpoint_error_handling=True, connect_timeout=5.0
):
    return await _get_ctx().create_endpoint(
        ip_address,
        port,
        endpoint_error_handling=endpoint_error_handling,
        connect_timeout=connect_timeout,
    )


async def create_endpoint_from_worker_address(
    address,
    endpoint_error_handling=True,
):
    return await _get_ctx().create_endpoint_from_worker_address(
        address,
        endpoint_error_handling=endpoint_error_handling,
    )


def get_ucp_context_info():
    """Gets information on the current UCX context, obtained from
    `ucp_context_print_info`.
    """
    return _get_ctx().ucp_context_info


def get_ucp_worker_info():
    """Gets information on the current UCX worker, obtained from
    `ucp_worker_print_info`.
    """
    return _get_ctx().ucp_worker_info


def get_active_transports():
    """Returns a list of all transports that are available and are currently
    active in UCX, meaning UCX **may** use them depending on the type of
    transfers and how it is configured but is not required to do so.
    """
    info = get_ucp_context_info()
    resources = re.findall("^#.*resource.*md.*dev.*flags.*$", info, re.MULTILINE)
    return set([r.split()[-1].split("/")[0] for r in resources])


def continuous_ucx_progress():
    _get_ctx().continuous_ucx_progress()


def get_ucp_worker():
    return _get_ctx().ucp_worker


def get_ucxx_worker():
    return _get_ctx().ucxx_worker


def get_worker_address():
    return _get_ctx().worker_address


def get_ucx_address_from_buffer(buffer):
    return ucx_api.UCXAddress.create_from_buffer(bytes(buffer))


async def recv(buffer, tag):
    return await _get_ctx().recv(buffer, tag=tag)


# Setting the __doc__
create_listener.__doc__ = ApplicationContext.create_listener.__doc__
create_endpoint.__doc__ = ApplicationContext.create_endpoint.__doc__
continuous_ucx_progress.__doc__ = ApplicationContext.continuous_ucx_progress.__doc__
get_ucp_worker.__doc__ = ApplicationContext.get_ucp_worker.__doc__
stop_notifier_thread.__doc__ = ApplicationContext.stop_notifier_thread.__doc__

__all__ = [
    "init",
    "reset",
    "stop_notifier_thread",
    "get_ucx_version",
    "progress",
    "get_config",
    "create_listener",
    "create_endpoint",
    "create_endpoint_from_worker_address",
    "get_ucp_context_info",
    "get_ucp_worker_info",
    "get_active_transports",
    "continuous_ucx_progress",
    "get_ucp_worker",
    "get_ucxx_worker",
    "get_worker_address",
    "get_ucx_address_from_buffer",
    "recv",
]
