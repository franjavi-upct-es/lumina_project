# tests/conftest.py
"""Shared pytest fixtures."""

import pytest
from dotenv import dotenv_values


@pytest.fixture
def anyio_backend():
    return "asyncio"
