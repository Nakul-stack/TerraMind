from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List


def detect_response_type(content_type: str, body: str) -> str:
    ct = (content_type or "").lower()
    if "json" in ct:
        return "json"
    if "xml" in ct or "rss" in ct:
        return "xml"
    if "html" in ct:
        return "html"

    b = (body or "").lstrip()
    if b.startswith("{") or b.startswith("["):
        return "json"
    if b.startswith("<"):
        return "xml" if "<rss" in b[:300].lower() else "html"
    return "text"


def parse_json_records(body: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    payload = json.loads(body)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                out.append(item)
        return out

    if isinstance(payload, dict):
        for key in ["results", "records", "items", "docs", "response"]:
            val = payload.get(key)
            if isinstance(val, list):
                out.extend([x for x in val if isinstance(x, dict)])
            elif isinstance(val, dict):
                nested_docs = val.get("docs")
                if isinstance(nested_docs, list):
                    out.extend([x for x in nested_docs if isinstance(x, dict)])
        if not out:
            out.append(payload)
    return out


def parse_xml_records(body: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    root = ET.fromstring(body)
    for item in root.findall(".//item") + root.findall(".//entry"):
        rec: Dict[str, Any] = {}
        title = item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title")
        summary = item.findtext("description") or item.findtext("summary") or item.findtext("{http://www.w3.org/2005/Atom}summary")
        link = item.findtext("link") or item.findtext("{http://www.w3.org/2005/Atom}link")
        rec["title"] = title or ""
        rec["abstract"] = summary or ""
        rec["url"] = link or ""
        out.append(rec)
    return out


def parse_html_result_cards(body: str) -> List[Dict[str, Any]]:
    text = re.sub(r"<script[\s\S]*?</script>", " ", body, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)

    cards = re.findall(r"<(article|div)[^>]*>([\s\S]*?)</\1>", text, flags=re.IGNORECASE)
    out: List[Dict[str, Any]] = []

    for _, card in cards[:80]:
        title_match = re.search(r"<h[1-4][^>]*>([\s\S]*?)</h[1-4]>", card, flags=re.IGNORECASE)
        link_match = re.search(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>([\s\S]*?)</a>", card, flags=re.IGNORECASE)
        abs_match = re.search(r"<p[^>]*>([\s\S]*?)</p>", card, flags=re.IGNORECASE)

        title = _strip_html(title_match.group(1)) if title_match else ""
        if not title and link_match:
            title = _strip_html(link_match.group(2))

        abstract = _strip_html(abs_match.group(1)) if abs_match else ""
        url = html.unescape(link_match.group(1)) if link_match else ""

        if len(title) >= 8:
            out.append({"title": title, "abstract": abstract, "url": url})

    if out:
        return out[:20]

    # fallback to loose heading/link extraction
    loose = []
    for heading in re.findall(r"<h[1-4][^>]*>([\s\S]*?)</h[1-4]>", text, flags=re.IGNORECASE)[:20]:
        cleaned = _strip_html(heading)
        if len(cleaned) > 8:
            loose.append({"title": cleaned, "abstract": "", "url": ""})
    return loose


def _strip_html(value: str) -> str:
    s = re.sub(r"<[^>]+>", " ", value or "")
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()
