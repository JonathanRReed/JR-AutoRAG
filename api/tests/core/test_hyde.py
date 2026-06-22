import pytest
from app.core.hyde import get_hyde_generator, HyDEConfig
import app.core.hyde as hyde_module

@pytest.fixture(autouse=True)
def reset_hyde_generator():
    """Reset the global HyDE generator before and after each test."""
    original_generator = hyde_module._hyde_generator
    hyde_module._hyde_generator = None
    yield
    hyde_module._hyde_generator = original_generator

def test_get_hyde_generator_singleton():
    """Test that get_hyde_generator returns a singleton instance unless config is provided."""

    # 1. First call should create a new instance
    gen1 = get_hyde_generator()
    assert gen1 is not None

    # 2. Second call without config should return the exact same instance
    gen2 = get_hyde_generator()
    assert gen1 is gen2

    # 3. Calling with a config should create a new instance and replace the singleton
    config = HyDEConfig(num_hypotheticals=5)
    gen3 = get_hyde_generator(config)
    assert gen3 is not gen1
    assert gen3.config.num_hypotheticals == 5

    # 4. Subsequent calls without config should return the new singleton
    gen4 = get_hyde_generator()
    assert gen4 is gen3
