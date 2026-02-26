"""Root conftest for quality-graft test suite."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="Run tests marked as 'heavy' (GPU, large checkpoints).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-heavy"):
        return
    skip_heavy = pytest.mark.skip(reason="needs --run-heavy option to run")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
