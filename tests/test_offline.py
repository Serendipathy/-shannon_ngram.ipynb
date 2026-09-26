"""AC 10 — the kill-switch is live, so the rest of the suite really is offline."""

from __future__ import annotations

import socket

import pytest

from conftest import NetworkAccessBlocked

from shannon_ngram.config import load_settings


def test_socket_is_blocked():
    with pytest.raises(NetworkAccessBlocked):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_create_connection_is_blocked():
    with pytest.raises(NetworkAccessBlocked):
        socket.create_connection(("api.ngrams.dev", 443), timeout=0.1)


def test_dns_is_blocked():
    with pytest.raises(NetworkAccessBlocked):
        socket.getaddrinfo("api.ngrams.dev", 443)


def test_a_real_client_cannot_reach_the_api():
    """A live NgramClient must fail, not silently succeed, under the kill-switch."""
    from shannon_ngram.client import NgramClient

    client = NgramClient(load_settings(max_retries=0, request_timeout=1.0))
    with pytest.raises(Exception) as excinfo:
        client.search("the cat *")
    assert not isinstance(excinfo.value, AssertionError)
    client.close()
