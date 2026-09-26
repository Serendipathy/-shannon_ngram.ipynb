"""AC 10 — the suite runs offline, and proves it.

Every test runs with ``socket.socket`` replaced by a stub that raises. An
accidental live call therefore fails loudly instead of quietly passing on a
machine that happens to have network.
"""

from __future__ import annotations

import socket

import pytest


class NetworkAccessBlocked(RuntimeError):
    pass


class BlockedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise NetworkAccessBlocked(
            "the test suite is offline by design; use a fixture or a fake transport"
        )


def _blocked(*args, **kwargs):
    raise NetworkAccessBlocked("network access is blocked during tests")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield
