from __future__ import annotations

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
    ],
)
def test_public_provider_url_rejects_non_global_addresses(url: str) -> None:
    assert is_public_provider_url(url) is False


def test_public_provider_url_allows_global_address() -> None:
    assert is_public_provider_url("https://8.8.8.8") is True
