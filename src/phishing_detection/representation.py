"""Parse email safely and build de-identified detector representations.

Version 1.0 contains the de-identified visible subject and body. Version 2.0
adds locally derived, inactive link and attachment structure. Neither version
performs network I/O or retains active URL values or attachment payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urlsplit

V1_VERSION = "detector-input-v1.0"
V2_VERSION = "detector-input-v2.0-links-attachments"
ACTIVE_URL_RE = re.compile(r"""(?i)\b(?:https?://|www\.)[^\s<>"']+""")
_URL_RE = ACTIVE_URL_RE
_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9.-]+\.[a-z]{2,}(?![\w-])")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d(). -]{7,}\d)(?!\w)")
_LONG_NUMBER_RE = re.compile(r"(?<!\w)\d{8,}(?!\w)")
_PATH_INTENTS = (
    "login",
    "signin",
    "verify",
    "account",
    "password",
    "payment",
    "invoice",
    "update",
)


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    subject: str
    body: str
    body_source: str
    has_plain: bool
    has_html: bool
    from_raw: str | None
    from_address: str | None
    from_domain: str | None
    to_raw: str | None
    to_addresses: tuple[str, ...]
    to_domains: tuple[str, ...]
    date_raw: str | None
    date_utc: str | None
    date_status: str
    message_id: str | None
    mime_type: str
    attachment_extensions: tuple[str, ...]
    warnings: tuple[str, ...]


class _VisibleText(HTMLParser):
    _hidden = {"head", "script", "style", "template", "svg"}
    _blocks = {
        "address",
        "article",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in self._hidden:
            self.depth += 1
        elif not self.depth and tag.casefold() in self._blocks:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in self._hidden and self.depth:
            self.depth -= 1
        elif not self.depth and tag.casefold() in self._blocks:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.depth:
            self.chunks.append(data)

    def text(self) -> str:
        lines = (
            re.sub(r"\s+", " ", line).strip()
            for line in "".join(self.chunks).splitlines()
        )
        return "\n".join(line for line in lines if line)


class _HrefCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        for name, value in attrs:
            if name.casefold() == "href" and value:
                self.targets.append(value.strip())


def _addresses(value: str | None) -> tuple[str, ...]:
    return tuple(
        address.strip().lower()
        for _, address in getaddresses([value or ""])
        if "@" in address
    )


def _domain(address: str | None) -> str | None:
    return address.rsplit("@", 1)[1].lower() if address and "@" in address else None


def parse_email(raw_email: bytes) -> ParsedMessage:
    """Parse one message without rendering HTML, opening attachments, or using a network."""
    message = BytesParser(policy=policy.default).parsebytes(raw_email)
    plain: list[str] = []
    html: list[str] = []
    extensions: list[str] = []
    warnings: list[str] = []
    for part in message.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            extensions.append(Path(part.get_filename() or "").suffix.lower())
            continue
        if part.get_content_type() not in {"text/plain", "text/html"}:
            continue
        content = part.get_content()
        if not isinstance(content, str) or not content.strip():
            continue
        (plain if part.get_content_type() == "text/plain" else html).append(
            content.strip()
        )
    visible_html: list[str] = []
    for value in html:
        parser = _VisibleText()
        parser.feed(value)
        parser.close()
        if parser.text():
            visible_html.append(parser.text())
    if plain:
        body, source = "\n\n".join(plain), "plain"
    elif visible_html:
        body, source = "\n\n".join(visible_html), "html_fallback"
    else:
        body, source = "", "missing"
    from_raw = str(message.get("From")) if message.get("From") is not None else None
    from_addresses = _addresses(from_raw)
    from_address = from_addresses[0] if from_addresses else None
    to_raw = str(message.get("To")) if message.get("To") is not None else None
    to_addresses = _addresses(to_raw)
    date_raw = str(message.get("Date")) if message.get("Date") is not None else None
    date_utc, date_status = None, "missing"
    if date_raw:
        try:
            parsed = parsedate_to_datetime(date_raw)
            date_utc, date_status = parsed.isoformat(), "ok"
        except (TypeError, ValueError, OverflowError):
            warnings.append("invalid_date")
            date_status = "invalid"
    message_id = str(message.get("Message-ID", "")).strip().strip("<>") or None
    return ParsedMessage(
        subject=str(message.get("Subject", "")).strip(),
        body=body,
        body_source=source,
        has_plain=bool(plain),
        has_html=bool(html),
        from_raw=from_raw,
        from_address=from_address,
        from_domain=_domain(from_address),
        to_raw=to_raw,
        to_addresses=to_addresses,
        to_domains=tuple(filter(None, map(_domain, to_addresses))),
        date_raw=date_raw,
        date_utc=date_utc,
        date_status=date_status,
        message_id=message_id,
        mime_type=message.get_content_type(),
        attachment_extensions=tuple(extensions),
        warnings=tuple(warnings),
    )


@lru_cache(maxsize=1)
def _ner() -> Any:
    import spacy

    return spacy.load("xx_ent_wiki_sm")


class _Deidentifier:
    def __init__(self, internal_domains: Iterable[str]) -> None:
        self.internal = {value.casefold() for value in internal_domains if value}
        self.maps: dict[str, dict[str, str]] = {
            key: {} for key in ("url", "email", "phone", "number", "person", "org")
        }

    def _token(self, kind: str, value: str, prefix: str | None = None) -> str:
        mapping = self.maps[kind]
        key = value.casefold()
        if key not in mapping:
            mapping[key] = f"[{prefix or kind.upper()}_{len(mapping) + 1}]"
        return mapping[key]

    def url(self, match: re.Match[str]) -> str:
        raw = match.group(0).rstrip(".,;!)]}")
        trailing = match.group(0)[len(raw) :]
        explicit = "://" in raw
        parsed = urlsplit(raw if explicit else f"http://{raw}")
        intent = next(
            (item for item in _PATH_INTENTS if item in parsed.path.casefold()), None
        )
        path = (
            f"[PATH_{intent.upper()}]"
            if intent
            else "[PATH_PRESENT]" if parsed.path not in {"", "/"} else "[PATH_ROOT]"
        )
        scheme = parsed.scheme.upper() if explicit else "NONE"
        domain = self._token("url", parsed.hostname or "unknown", "URL_DOMAIN")
        query = "[QUERY_REMOVED]" if parsed.query else ""
        fragment = "[FRAGMENT_REMOVED]" if parsed.fragment else ""
        return f"[URL_SCHEME_{scheme}]{domain}{path}{query}{fragment}{trailing}"

    def transform(self, text: str) -> str:
        text = _URL_RE.sub(self.url, text)
        text = _EMAIL_RE.sub(lambda m: self._token("email", m.group(0), "EMAIL"), text)
        text = _PHONE_RE.sub(
            lambda m: (
                self._token("phone", re.sub(r"\D", "", m.group(0)), "PHONE")
                if m.group(0).startswith("+") or re.search(r"[ ().-]", m.group(0))
                else m.group(0)
            ),
            text,
        )
        text = _LONG_NUMBER_RE.sub(
            lambda m: self._token("number", m.group(0), "LONG_NUMBER"), text
        )
        document = _ner()(text)
        replacements: list[tuple[int, int, str]] = []
        for entity in document.ents:
            kind = (
                "person"
                if entity.label_ in {"PER", "PERSON"}
                else "org" if entity.label_ == "ORG" else None
            )
            if (
                kind
                and "[" not in entity.text
                and any(char.isalpha() for char in entity.text)
            ):
                replacements.append(
                    (entity.start_char, entity.end_char, self._token(kind, entity.text))
                )
        for start, end, replacement in reversed(replacements):
            text = text[:start] + replacement + text[end:]
        return text


def deidentify(
    subject: str,
    body: str,
    *,
    sender_domain: str | None,
    recipient_domains: Iterable[str],
) -> tuple[str, str, str]:
    state = _Deidentifier([sender_domain or "", *recipient_domains])
    safe_subject = state.transform(subject)
    safe_body = state.transform(body)
    return safe_subject, safe_body, f"Subject: {safe_subject}\n\n{safe_body}"


def enrich_v2(
    v1_text: str, raw_email: bytes, attachment_extensions: Iterable[str]
) -> str:
    """Append inactive structure while discarding every destination and payload value."""
    attachment_extensions = tuple(attachment_extensions)
    message = BytesParser(policy=policy.default).parsebytes(raw_email)
    targets: list[str] = []
    for part in message.walk():
        if part.is_multipart() or part.get_content_type() != "text/html":
            continue
        content = part.get_content()
        if isinstance(content, str):
            collector = _HrefCollector()
            collector.feed(content)
            collector.close()
            targets.extend(collector.targets)
    domains: dict[str, str] = {}
    structures: list[str] = []
    for target in targets:
        value = f"http://{target}" if target.casefold().startswith("www.") else target
        parsed = urlsplit(value)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            continue
        host = parsed.hostname.casefold()
        domains.setdefault(host, f"[HTML_LINK_DOMAIN_{len(domains) + 1}]")
        intent = next(
            (item for item in _PATH_INTENTS if item in parsed.path.casefold()), None
        )
        path = (
            f"[HTML_LINK_PATH_{intent.upper()}]"
            if intent
            else (
                "[HTML_LINK_PATH_PRESENT]"
                if parsed.path not in {"", "/"}
                else "[HTML_LINK_PATH_ROOT]"
            )
        )
        query = "[HTML_LINK_QUERY_REMOVED]" if parsed.query else ""
        fragment = "[HTML_LINK_FRAGMENT_REMOVED]" if parsed.fragment else ""
        structures.append(
            f"[HTML_LINK_SCHEME_{parsed.scheme.upper()}]"
            f"{domains[host]}{path}{query}{fragment}"
        )
    extension_tokens = sorted(
        {
            (
                f"[ATTACHMENT_EXT_{ext.lstrip('.').upper()}]"
                if re.fullmatch(r"[A-Za-z0-9]{1,10}", ext.lstrip("."))
                else "[ATTACHMENT_EXT_OTHER]"
            )
            for ext in attachment_extensions
        }
    )
    result = "\n".join(
        (
            v1_text,
            "",
            "Local structural metadata (values deactivated):",
            f"HTML link count: {len(structures)}",
            "HTML link structures: "
            + (" ".join(dict.fromkeys(structures)) or "[NONE]"),
            f"Attachment count: {len(attachment_extensions)}",
            "Attachment types: " + (" ".join(extension_tokens) or "[NONE]"),
        )
    )
    if ACTIVE_URL_RE.search(result):
        raise ValueError("Detector Input v2.0 contains an active URL")
    return result
