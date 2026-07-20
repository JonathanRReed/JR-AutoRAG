from __future__ import annotations

import socket

import pytest

from app.core.providers import ProviderError, discover_models
from app.schemas.config import ProviderConfig, is_public_provider_url


@pytest.mark.asyncio
async def test_local_provider_discovery_rejects_private_network_url() -> None:
    cfg = ProviderConfig(name="lm", base_url="http://10.0.0.5:8080")

    with pytest.raises(ProviderError, match="localhost or loopback"):
        await discover_models(cfg)


@pytest.mark.parametrize(
    "url",
    [
        "http://100.64.0.1",
        "http://100.127.255.254",
        "http://169.254.1.1",
        "http://192.168.1.10",
        "http://224.0.0.1",
        "http://239.255.255.250",
        "http://[ff02::1]",
        "http://[64:ff9b::1]",
    ],
)
def test_public_provider_url_rejects_non_global_addresses(url: str) -> None:
    assert is_public_provider_url(url) is False


def test_public_provider_url_rejects_dns_multicast(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(*args: object, **kwargs: object) -> list[tuple[object, object, object, object, tuple[str, int]]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("239.255.255.250", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert is_public_provider_url("https://models.example.test") is False


def test_public_provider_url_allows_global_address() -> None:
    assert is_public_provider_url("https://8.8.8.8") is True
