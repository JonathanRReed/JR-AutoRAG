from app.core.multimodal import MultimodalProcessor
from app.core.tools import CalculatorTool


def test_calculator_evaluates_basic_arithmetic() -> None:
    result = CalculatorTool().execute(expression="(12 + 8) / 5")

    assert result.success is True
    assert result.result == 4


def test_calculator_handles_percent_values() -> None:
    result = CalculatorTool().execute(expression="50% + 0.25")

    assert result.success is True
    assert result.result == 0.75


def test_calculator_rejects_code_injection() -> None:
    result = CalculatorTool().execute(expression="__import__('os').system('id')")

    assert result.success is False
    assert "unsupported" in (result.error or "").lower()


def test_calculator_rejects_exponent_dos() -> None:
    result = CalculatorTool().execute(expression="2 ** 100000")

    assert result.success is False
    assert "unsupported" in (result.error or "").lower()


def test_multimodal_processor_marks_missing_vision_caption_backend() -> None:
    processor = MultimodalProcessor()
    images = processor.process_document("![Architecture diagram](diagram.png)")

    assert len(images) == 1
    assert images[0].description == ""
    assert images[0].metadata["vision_description_status"] == "not_configured"
