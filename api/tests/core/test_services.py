from unittest.mock import MagicMock, patch

from app.schemas.config import AppConfig, DeploymentProfile, ProviderConfig
from app.services import ServiceContainer


def test_prepare_config_for_storage() -> None:
    container = object.__new__(ServiceContainer)

    existing_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key="old-key"
        )
    )
    container.config_store = MagicMock()
    container.config_store.read.return_value = existing_cfg

    new_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key="new-key"
        )
    )

    with patch("app.core.providers._infer_secret_key_name", return_value="test_key"), \
         patch("app.core.secrets_vault.get_secrets_vault") as mock_get_vault, \
         patch("app.state.set_orchestrator"):

        mock_vault = MagicMock()
        mock_get_vault.return_value = mock_vault
        mock_vault.set.return_value = None

        result = container.prepare_config_for_storage(new_cfg)

        container.config_store.read.assert_called_once()
        assert result.provider is not None
        assert result.provider.api_key is None
        mock_vault.set.assert_called_with("test_key", "new-key")

def test_prepare_config_for_storage_handles_no_secret() -> None:
    container = object.__new__(ServiceContainer)

    existing_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=None
        )
    )
    container.config_store = MagicMock()
    container.config_store.read.return_value = existing_cfg

    new_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=None
        )
    )

    with patch("app.core.providers._infer_secret_key_name", return_value="test_key"), \
         patch("app.core.secrets_vault.get_secrets_vault") as mock_get_vault, \
         patch("app.state.set_orchestrator"):

        mock_vault = MagicMock()
        mock_get_vault.return_value = mock_vault
        # simulate the key is not in the vault either
        mock_vault.get.return_value = None

        result = container.prepare_config_for_storage(new_cfg)

        container.config_store.read.assert_called_once()
        assert result.provider is not None
        assert result.provider.api_key is None
        mock_vault.set.assert_not_called()

def test_prepare_config_for_storage_vault_failure() -> None:
    container = object.__new__(ServiceContainer)

    existing_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=None
        )
    )
    container.config_store = MagicMock()
    container.config_store.read.return_value = existing_cfg

    new_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key="new-key-that-fails-to-store"
        )
    )

    with patch("app.core.providers._infer_secret_key_name", return_value="test_key"), \
         patch("app.core.secrets_vault.get_secrets_vault") as mock_get_vault, \
         patch("app.state.set_orchestrator"):

        mock_vault = MagicMock()
        mock_get_vault.return_value = mock_vault
        mock_vault.set.side_effect = Exception("Vault unavailable")

        result = container.prepare_config_for_storage(new_cfg)

        container.config_store.read.assert_called_once()
        assert result.provider is not None
        # If vault storage fails, the config should retain the key rather than stripping it?
        # Let's see what it does.
        # Actually sanitize_provider says:
        # if store_secret(key_name, candidate):
        #     return provider.model_copy(update={"api_key": None})
        # return provider
        assert result.provider.api_key == "new-key-that-fails-to-store"
        mock_vault.set.assert_called_with("test_key", "new-key-that-fails-to-store")

def test_prepare_config_for_storage_uses_fallback() -> None:
    container = object.__new__(ServiceContainer)

    existing_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key="old-key-fallback"
        )
    )
    container.config_store = MagicMock()
    container.config_store.read.return_value = existing_cfg

    new_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            api_key=None
        )
    )

    with patch("app.core.providers._infer_secret_key_name", return_value="test_key"), \
         patch("app.core.secrets_vault.get_secrets_vault") as mock_get_vault, \
         patch("app.state.set_orchestrator"):

        mock_vault = MagicMock()
        mock_get_vault.return_value = mock_vault
        mock_vault.set.return_value = None

        result = container.prepare_config_for_storage(new_cfg)

        container.config_store.read.assert_called_once()
        assert result.provider is not None
        assert result.provider.api_key is None
        mock_vault.set.assert_called_with("test_key", "old-key-fallback")

def test_prepare_config_for_storage_handles_no_provider() -> None:
    container = object.__new__(ServiceContainer)

    existing_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=None
    )
    container.config_store = MagicMock()
    container.config_store.read.return_value = existing_cfg

    new_cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=None
    )

    with patch("app.core.secrets_vault.get_secrets_vault") as mock_get_vault, \
         patch("app.state.set_orchestrator"):

        mock_vault = MagicMock()
        mock_get_vault.return_value = mock_vault

        result = container.prepare_config_for_storage(new_cfg)

        container.config_store.read.assert_called_once()
        assert result.provider is None
        mock_vault.set.assert_not_called()
