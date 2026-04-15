from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List

import requests

from ..parsers import detect_response_type, parse_html_result_cards, parse_json_records, parse_xml_records
from ..types import QueryProfile, SourceCallLog


@dataclass
class AdapterCapability:
    source: str
    access_type: str
    expected_result_type: str
    full_text_likely: bool
    metadata_only_likely: bool
    reliability: str
    notes: str
    source_group: str = "research"
    enrichment_only: bool = False


class SourceAdapter(abc.ABC):
    source_name: str = "unknown"
    timeout_seconds: float = 12.0

    @abc.abstractmethod
    def capability(self) -> AdapterCapability:
        raise NotImplementedError

    @abc.abstractmethod
    def build_requests(self, profile: QueryProfile) -> List[Dict]:
        raise NotImplementedError

    def search(self, profile: QueryProfile) -> (List[Dict], List[SourceCallLog]):
        records: List[Dict] = []
        logs: List[SourceCallLog] = []

        for req in self.build_requests(profile):
            call = SourceCallLog(
                source=self.source_name,
                query=req.get("query", ""),
                url=req.get("url", ""),
                method=req.get("method", "GET"),
                payload=req.get("params", {}),
            )
            try:
                resp = requests.request(
                    method=req.get("method", "GET"),
                    url=req.get("url"),
                    params=req.get("params"),
                    headers=req.get("headers"),
                    timeout=req.get("timeout", self.timeout_seconds),
                )
                call.status_code = resp.status_code
                call.raw_sample = (resp.text or "")[:900]

                if resp.status_code in {401, 403, 429}:
                    call.blocked_error = True
                    logs.append(call)
                    continue

                response_type = detect_response_type(resp.headers.get("content-type", ""), resp.text)
                call.response_type = response_type

                parsed = []
                if response_type == "json":
                    parsed = parse_json_records(resp.text)
                elif response_type == "xml":
                    parsed = parse_xml_records(resp.text)
                elif response_type == "html":
                    parsed = parse_html_result_cards(resp.text)

                call.parsed_item_count = len(parsed)
                call.preview_items = parsed[:3]
                records.extend(parsed)
            except requests.Timeout:
                call.timeout_error = True
            except Exception as exc:
                call.other_error = str(exc)
            logs.append(call)

        return records, logs
