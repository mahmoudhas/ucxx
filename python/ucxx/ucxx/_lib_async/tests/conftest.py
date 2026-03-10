# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import inspect
import os

import pytest
import trio

import ucxx

os.environ["RAPIDS_NO_INITIALIZE"] = "True"


def pytest_runtest_teardown(item, nextitem):
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def ucxx_setup_teardown(request):
    """Reset UCX before/after every test.

    For async tests, wraps the test function so that a trio nursery is
    opened and stored in ``ucxx.core._test_nursery`` before the test
    body runs.  This lets ``_get_ctx()`` lazily initialise UCX even when
    the test doesn't call ``ucxx.init()`` explicitly.
    """
    ucxx.reset()

    original = getattr(request.node, "obj", None)
    if original is not None and inspect.iscoroutinefunction(original):
        async def _with_nursery(*args, **kwargs):
            async with trio.open_nursery() as nursery:
                ucxx.core._test_nursery = nursery
                try:
                    return await original(*args, **kwargs)
                finally:
                    ucxx.reset()
                    ucxx.core._test_nursery = None
                    nursery.cancel_scope.cancel()

        _with_nursery.__name__ = original.__name__
        _with_nursery.__module__ = original.__module__
        # Carry forward pytest markers
        _with_nursery.pytestmark = getattr(original, "pytestmark", [])
        request.node.obj = _with_nursery

    yield
    ucxx.reset()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    """Store test timeout for internal helpers."""
    timeout_marker = pyfuncitem.get_closest_marker("trio_timeout")
    if timeout_marker is None:
        timeout_marker = pyfuncitem.get_closest_marker("asyncio_timeout")
    slow_marker = pyfuncitem.get_closest_marker("slow")
    default_timeout = 600.0 if slow_marker else 60.0
    timeout = float(timeout_marker.args[0]) if timeout_marker else default_timeout
    pyfuncitem.config.cache.set("trio_test_timeout", {"timeout": timeout})
    yield


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "trio_timeout(timeout): cancels the test execution after the specified "
        "number of seconds",
    )
    config.addinivalue_line(
        "markers",
        "asyncio_timeout(timeout): alias for trio_timeout (migration convenience)",
    )
    config.addinivalue_line(
        "markers",
        "rerun_on_failure(reruns): reruns test if it fails for the specified number "
        "of reruns",
    )
    config.addinivalue_line("markers", "slow: mark test as slow to run")
