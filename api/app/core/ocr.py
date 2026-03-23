"""OCR routing and confidence-scored local-first extraction."""

from __future__ import annotations

import base64
import io
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx

try:  # pragma: no cover
    from pdf2image import convert_from_bytes  # type: ignore
except ImportError:  # pragma: no cover
    convert_from_bytes = None  # type: ignore

try:  # pragma: no cover
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore

from ..schemas.config import OCRPolicy, OCRSettings, ProviderConfig
from .providers import resolve_provider_api_key


@dataclass
class OCRResult:
    text: str
    method: str
    engine: str
    confidence: float
    used_ocr: bool
    attempted: list[str]


class BaseOCRProvider:
    backend_id: str = ""
    engine: str = ""

    def available(self) -> bool:
        raise NotImplementedError

    def extract(self, content: bytes) -> OCRResult:
        raise NotImplementedError


class TesseractOCRProvider(BaseOCRProvider):
    backend_id = "ocr.local.tesseract"
    engine = "tesseract"

    def available(self) -> bool:
        return convert_from_bytes is not None and pytesseract is not None

    def extract(self, content: bytes) -> OCRResult:
        if not self.available():
            return OCRResult("", "ocr_unavailable", self.engine, 0.0, False, [self.backend_id])

        poppler_bin = shutil.which("pdftoppm") or shutil.which("pdftocairo")
        poppler_path = str(Path(poppler_bin).parent) if poppler_bin else None
        tesseract_bin = shutil.which("tesseract")
        if tesseract_bin:
            pytesseract.pytesseract.tesseract_cmd = tesseract_bin  # type: ignore[attr-defined]

        try:
            images = convert_from_bytes(content, poppler_path=poppler_path)  # type: ignore[name-defined]
        except Exception:
            return OCRResult("", "ocr_error", self.engine, 0.0, False, [self.backend_id])

        text_chunks: list[str] = []
        total_confidence = 0.0
        confidence_samples = 0
        for image in images:
            try:
                text = pytesseract.image_to_string(image)  # type: ignore[attr-defined]
                if text:
                    text_chunks.append(text)
                try:
                    data = pytesseract.image_to_data(  # type: ignore[attr-defined]
                        image,
                        output_type=pytesseract.Output.DICT,  # type: ignore[attr-defined]
                    )
                    confidences = [
                        float(raw)
                        for raw in data.get("conf", [])
                        if raw not in {"-1", "", None}
                    ]
                    if confidences:
                        total_confidence += max(0.0, min(sum(confidences) / len(confidences), 100.0)) / 100.0
                        confidence_samples += 1
                except Exception:
                    pass
            finally:
                image.close()

        confidence = total_confidence / confidence_samples if confidence_samples else _confidence_from_text("\n".join(text_chunks))
        return OCRResult(
            text="\n".join(text_chunks),
            method="dedicated_ocr",
            engine=self.engine,
            confidence=confidence,
            used_ocr=bool(text_chunks),
            attempted=[self.backend_id],
        )


class VisionModelOCRProvider(BaseOCRProvider):
    backend_id = "ocr.local.vision"
    engine = "vision_model"

    def __init__(
        self,
        provider_config: ProviderConfig | None = None,
        *,
        model: str | None = None,
        max_pages: int = 8,
    ) -> None:
        self._provider_config = provider_config
        self._model = model
        self._max_pages = max_pages

    def available(self) -> bool:
        return convert_from_bytes is not None and self._provider_config is not None and bool(self._resolved_model())

    def extract(self, content: bytes) -> OCRResult:
        if not self.available():
            return OCRResult("", "ocr_unavailable", self.engine, 0.0, False, [self.backend_id])

        provider = self._provider_config
        if provider is None:
            return OCRResult("", "ocr_unavailable", self.engine, 0.0, False, [self.backend_id])

        poppler_bin = shutil.which("pdftoppm") or shutil.which("pdftocairo")
        poppler_path = str(Path(poppler_bin).parent) if poppler_bin else None
        try:
            images = convert_from_bytes(content, poppler_path=poppler_path)  # type: ignore[name-defined]
        except Exception:
            return OCRResult("", "ocr_error", self.engine, 0.0, False, [self.backend_id])

        text_chunks: list[str] = []
        confidence_scores: list[float] = []
        model = self._resolved_model()
        api_key = resolve_provider_api_key(provider.name, str(provider.base_url), provider.api_key)
        endpoint = self._resolve_chat_endpoint(str(provider.base_url))
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        client = httpx.Client(timeout=90.0, headers=headers)
        try:
            for image in images[: self._max_pages]:
                try:
                    image_payload = self._image_to_data_url(image)
                    prompt = (
                        "Transcribe all readable text from this page exactly as text only. "
                        "Preserve headings, lists, and table-like rows when possible. "
                        "Do not add commentary."
                    )
                    response = client.post(
                        endpoint,
                        json={
                            "model": model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": image_payload},
                                    ],
                                }
                            ],
                            "stream": False,
                            "temperature": 0,
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                    message = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                    if isinstance(message, list):
                        parts: list[str] = []
                        for item in message:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts.append(str(item.get("text", "")))
                            else:
                                parts.append(str(item))
                        message = "\n".join(part for part in parts if part)
                    extracted = str(message).strip()
                    if extracted:
                        text_chunks.append(extracted)
                        confidence_scores.append(_confidence_from_text(extracted))
                except Exception:
                    continue
                finally:
                    image.close()
            for image in images[self._max_pages :]:
                try:
                    image.close()
                except Exception:
                    continue
        finally:
            client.close()

        combined = "\n\n".join(text_chunks)
        confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        return OCRResult(
            text=combined,
            method="vision_model",
            engine=f"{self.engine}:{model}",
            confidence=confidence,
            used_ocr=bool(combined),
            attempted=[self.backend_id],
        )

    def _resolved_model(self) -> str | None:
        if self._model:
            return self._model
        if self._provider_config is None:
            return None
        return (
            self._provider_config.generator_model
            or self._provider_config.gatherer_model
            or self._provider_config.planner_model
        )

    def _resolve_chat_endpoint(self, base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _image_to_data_url(self, image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


def _confidence_from_text(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    alpha = sum(1 for ch in stripped if ch.isalpha())
    printable = sum(1 for ch in stripped if ch.isprintable() and not ch.isspace())
    density = min(len(stripped) / 400.0, 1.0)
    alpha_ratio = alpha / max(printable, 1)
    return max(0.0, min((density * 0.5) + (alpha_ratio * 0.5), 1.0))


class OCRRouter:
    def __init__(
        self,
        settings: OCRSettings,
        *,
        provider_config: ProviderConfig | None = None,
        vision_model: str | None = None,
        vision_max_pages: int = 8,
    ) -> None:
        self._settings = settings
        self._providers = {
            "ocr.local.tesseract": TesseractOCRProvider(),
            "ocr.local.vision": VisionModelOCRProvider(
                provider_config,
                model=vision_model,
                max_pages=vision_max_pages,
            ),
        }

    def score_extractable_text(self, text: str) -> float:
        stripped = text.strip()
        if len(stripped) < self._settings.min_characters:
            return _confidence_from_text(stripped) * 0.5
        return _confidence_from_text(stripped)

    def route(self, extracted_text: str, content: bytes) -> OCRResult:
        extract_confidence = self.score_extractable_text(extracted_text)
        attempted = ["native_text"]
        if self._settings.policy == OCRPolicy.OFF:
            return OCRResult(
                text=extracted_text,
                method="native_text",
                engine="text_parser",
                confidence=extract_confidence,
                used_ocr=False,
                attempted=attempted,
            )

        if extracted_text.strip() and extract_confidence >= self._settings.extractable_text_threshold:
            return OCRResult(
                text=extracted_text,
                method="native_text",
                engine="text_parser",
                confidence=extract_confidence,
                used_ocr=False,
                attempted=attempted,
            )

        if self._settings.policy == OCRPolicy.DEDICATED_OCR:
            result = self._run_provider("ocr.local.tesseract", content, attempted)
            return result if result.text.strip() else self._native_fallback(extracted_text, extract_confidence, attempted)

        if self._settings.policy == OCRPolicy.VISION_MODEL:
            result = self._run_provider("ocr.local.vision", content, attempted)
            return result if result.text.strip() else self._native_fallback(extracted_text, extract_confidence, attempted)

        if self._settings.policy == OCRPolicy.HYBRID:
            return self._run_hybrid(content, extracted_text, extract_confidence, attempted)

        for backend_id in self._settings.preferred_backends:
            result = self._run_provider(backend_id, content, attempted)
            if result.text.strip():
                return result

        return self._native_fallback(extracted_text, extract_confidence, attempted)

    def _native_fallback(self, extracted_text: str, confidence: float, attempted: list[str]) -> OCRResult:
        return OCRResult(
            text=extracted_text,
            method="native_text_fallback",
            engine="text_parser",
            confidence=confidence,
            used_ocr=False,
            attempted=attempted,
        )

    def _run_provider(self, backend_id: str, content: bytes, attempted: list[str]) -> OCRResult:
        attempted.append(backend_id)
        provider = self._providers.get(backend_id)
        if provider is None:
            return OCRResult("", "ocr_unavailable", backend_id, 0.0, False, attempted)
        result = provider.extract(content)
        result.attempted = list(dict.fromkeys(attempted + result.attempted))
        return result

    def _run_hybrid(
        self,
        content: bytes,
        extracted_text: str,
        extract_confidence: float,
        attempted: list[str],
    ) -> OCRResult:
        tesseract_result = self._run_provider("ocr.local.tesseract", content, attempted)
        vision_result = self._run_provider("ocr.local.vision", content, attempted)
        candidates = [tesseract_result, vision_result]
        candidates = [item for item in candidates if item.text.strip()]
        if not candidates:
            return self._native_fallback(extracted_text, extract_confidence, attempted)

        best = max(candidates, key=lambda item: item.confidence)
        if self._settings.dual_merge_strategy == "prefer_text_parser" and extract_confidence >= best.confidence:
            return self._native_fallback(extracted_text, extract_confidence, attempted)
        return OCRResult(
            text=best.text,
            method="hybrid_ocr",
            engine=best.engine,
            confidence=best.confidence,
            used_ocr=True,
            attempted=list(dict.fromkeys(attempted + best.attempted)),
        )
