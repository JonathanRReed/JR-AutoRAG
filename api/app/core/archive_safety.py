"""Bounded validation for archive-based document formats."""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath

MAX_ARCHIVE_MEMBERS = 2048
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 200
XML_SCAN_CHUNK_BYTES = 64 * 1024
FORBIDDEN_XML_MARKERS = (b"<!DOCTYPE", b"<!ENTITY")


class UnsafeArchiveError(ValueError):
    """Raised when an uploaded archive exceeds safe processing limits."""


def _contains_forbidden_xml_declaration(xml_file) -> bool:
    carry = b""
    carry_size = max(len(marker) for marker in FORBIDDEN_XML_MARKERS) - 1
    while chunk := xml_file.read(XML_SCAN_CHUNK_BYTES):
        scanned = carry + chunk.upper()
        if any(marker in scanned for marker in FORBIDDEN_XML_MARKERS):
            return True
        carry = scanned[-carry_size:]
    return False


def validate_docx_archive(content: bytes) -> None:
    """Reject malformed or resource-intensive DOCX containers before parsing."""

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise UnsafeArchiveError(
                    f"DOCX archive contains too many members ({len(members)} > {MAX_ARCHIVE_MEMBERS})."
                )

            names: set[str] = set()
            total_uncompressed = 0
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise UnsafeArchiveError(
                        "DOCX archive contains an unsafe member path."
                    )
                if member.filename in names:
                    raise UnsafeArchiveError(
                        "DOCX archive contains duplicate member names."
                    )
                names.add(member.filename)
                if member.flag_bits & 0x1:
                    raise UnsafeArchiveError(
                        "Encrypted DOCX archive members are not supported."
                    )
                if member.file_size < 0 or member.compress_size < 0:
                    raise UnsafeArchiveError(
                        "DOCX archive contains invalid member sizes."
                    )
                if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise UnsafeArchiveError(
                        "DOCX archive member exceeds the uncompressed size limit."
                    )

                total_uncompressed += member.file_size
                if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise UnsafeArchiveError(
                        "DOCX archive exceeds the total uncompressed size limit."
                    )

                if member.file_size:
                    ratio = member.file_size / max(member.compress_size, 1)
                    if ratio > MAX_ARCHIVE_COMPRESSION_RATIO:
                        raise UnsafeArchiveError(
                            "DOCX archive member exceeds the safe compression ratio."
                        )

            if "word/document.xml" not in names:
                raise UnsafeArchiveError("DOCX archive is missing word/document.xml.")

            for member in members:
                if not member.filename.lower().endswith((".xml", ".rels")):
                    continue
                with archive.open(member) as xml_file:
                    if _contains_forbidden_xml_declaration(xml_file):
                        raise UnsafeArchiveError(
                            "DOCX archive contains a forbidden XML declaration."
                        )
    except UnsafeArchiveError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise UnsafeArchiveError("DOCX archive is malformed or unreadable.") from exc


__all__ = ["UnsafeArchiveError", "validate_docx_archive"]
