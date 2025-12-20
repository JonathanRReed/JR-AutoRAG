"""Multimodal document processing for images and diagrams.

This module provides a framework for:
- Image detection in documents
- OCR text extraction (when available)
- Image captioning/description (placeholder for vision models)
- Diagram interpretation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ImageType(str, Enum):
    """Types of images in documents."""
    PHOTO = "photo"
    DIAGRAM = "diagram"
    CHART = "chart"
    SCREENSHOT = "screenshot"
    ICON = "icon"
    UNKNOWN = "unknown"


@dataclass
class ExtractedImage:
    """Information about an extracted image."""
    image_id: str
    source_path: str | None
    image_type: ImageType
    position: int  # Position in document
    alt_text: str
    caption: str
    ocr_text: str  # Text extracted via OCR
    description: str  # AI-generated description
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_text_representation(self) -> str:
        """Convert to searchable text."""
        parts = []
        
        if self.caption:
            parts.append(f"Image: {self.caption}")
        if self.alt_text:
            parts.append(f"Description: {self.alt_text}")
        if self.ocr_text:
            parts.append(f"Text in image: {self.ocr_text}")
        if self.description:
            parts.append(self.description)
        
        return " ".join(parts) if parts else "[Image]"


class ImageExtractor:
    """Extracts image references from documents."""
    
    # Markdown image pattern
    MD_IMAGE_PATTERN = re.compile(
        r'!\[([^\]]*)\]\(([^)]+)\)',
        re.MULTILINE
    )
    
    # HTML image pattern
    HTML_IMAGE_PATTERN = re.compile(
        r'<img[^>]+src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?[^>]*>',
        re.IGNORECASE
    )
    
    def extract_markdown_images(self, text: str) -> list[tuple[str, str, int]]:
        """Extract markdown image references.
        
        Returns:
            List of (alt_text, src, position)
        """
        images = []
        for match in self.MD_IMAGE_PATTERN.finditer(text):
            alt_text = match.group(1)
            src = match.group(2)
            position = match.start()
            images.append((alt_text, src, position))
        return images
    
    def extract_html_images(self, text: str) -> list[tuple[str, str, int]]:
        """Extract HTML image references.
        
        Returns:
            List of (alt_text, src, position)
        """
        images = []
        for match in self.HTML_IMAGE_PATTERN.finditer(text):
            src = match.group(1)
            alt_text = match.group(2) or ""
            position = match.start()
            images.append((alt_text, src, position))
        return images
    
    def extract_all(self, text: str) -> list[tuple[str, str, int]]:
        """Extract all image references."""
        images = []
        images.extend(self.extract_markdown_images(text))
        images.extend(self.extract_html_images(text))
        # Sort by position
        images.sort(key=lambda x: x[2])
        return images


class OCRProcessor:
    """OCR text extraction from images.
    
    This is a placeholder that can be extended with actual OCR.
    The system already has pytesseract for PDF OCR.
    """
    
    def __init__(self) -> None:
        self._tesseract_available = False
        self._check_tesseract()
    
    def _check_tesseract(self) -> None:
        """Check if tesseract is available."""
        try:
            import pytesseract
            # Quick check
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception:
            self._tesseract_available = False
    
    @property
    def available(self) -> bool:
        return self._tesseract_available
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        if not self._tesseract_available:
            return ""
        
        try:
            import pytesseract
            from PIL import Image
            
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception:
            return ""


class ImageTypeClassifier:
    """Classifies images into types based on heuristics."""
    
    # File extension hints
    EXTENSION_HINTS = {
        '.png': ImageType.SCREENSHOT,
        '.jpg': ImageType.PHOTO,
        '.jpeg': ImageType.PHOTO,
        '.gif': ImageType.ICON,
        '.svg': ImageType.DIAGRAM,
    }
    
    # Filename pattern hints
    FILENAME_PATTERNS = {
        r'diagram|flowchart|chart|graph': ImageType.DIAGRAM,
        r'screenshot|screen|capture': ImageType.SCREENSHOT,
        r'icon|logo|badge': ImageType.ICON,
        r'photo|image|img': ImageType.PHOTO,
    }
    
    def classify(self, src: str, alt_text: str = "") -> ImageType:
        """Classify image type based on available information."""
        combined = f"{src} {alt_text}".lower()
        
        # Check filename patterns
        for pattern, img_type in self.FILENAME_PATTERNS.items():
            if re.search(pattern, combined):
                return img_type
        
        # Check extension
        path = Path(src)
        ext = path.suffix.lower()
        if ext in self.EXTENSION_HINTS:
            return self.EXTENSION_HINTS[ext]
        
        return ImageType.UNKNOWN


class MultimodalProcessor:
    """Main processor for multimodal document content."""
    
    def __init__(self) -> None:
        self.image_extractor = ImageExtractor()
        self.ocr_processor = OCRProcessor()
        self.classifier = ImageTypeClassifier()
    
    def process_document(
        self,
        text: str,
        base_path: str | None = None,
    ) -> list[ExtractedImage]:
        """Process document and extract image information.
        
        Args:
            text: Document text content
            base_path: Base path for resolving relative image paths
        
        Returns:
            List of ExtractedImage with available information
        """
        images = []
        
        for i, (alt_text, src, position) in enumerate(self.image_extractor.extract_all(text)):
            image_id = f"img_{i}"
            
            # Resolve path if base provided
            image_path = None
            if base_path and not src.startswith(('http://', 'https://', 'data:')):
                full_path = Path(base_path) / src
                if full_path.exists():
                    image_path = str(full_path)
            
            # Classify image type
            image_type = self.classifier.classify(src, alt_text)
            
            # Extract OCR text if path available
            ocr_text = ""
            if image_path and self.ocr_processor.available:
                ocr_text = self.ocr_processor.extract_text(image_path)
            
            images.append(ExtractedImage(
                image_id=image_id,
                source_path=image_path,
                image_type=image_type,
                position=position,
                alt_text=alt_text,
                caption=alt_text,  # Use alt as caption if none provided
                ocr_text=ocr_text,
                description="",  # Would be filled by vision model
            ))
        
        return images
    
    def enhance_text_with_images(
        self,
        text: str,
        images: list[ExtractedImage],
    ) -> str:
        """Enhance document text with image descriptions for retrieval."""
        enhanced_parts = [text]
        
        for img in images:
            representation = img.to_text_representation()
            if representation and representation != "[Image]":
                enhanced_parts.append(f"\n[Image Content: {representation}]")
        
        return "\n".join(enhanced_parts)


__all__ = [
    "ImageType",
    "ExtractedImage",
    "ImageExtractor",
    "OCRProcessor",
    "ImageTypeClassifier",
    "MultimodalProcessor",
]
