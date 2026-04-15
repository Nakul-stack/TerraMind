import json
import os
import re
import time
from dataclasses import asdict
from typing import Dict, Optional

import requests

from .graph_builder import AgroKGBuilder
from .intent_parser import IntentParser, ParsedIntent
from .query_engine import GraphQueryEngine
from .retrieval import ExternalRetrievalOrchestrator


class GraphRAGPipeline:
    """End-to-end GraphRAG pipeline: parse -> query KG -> generate answer."""

    _AGRI_DOMAIN_PATTERNS = [
        r"\bcrop(s)?\b",
        r"\bsoil\b|\bsoil science\b",
        r"\birrigation\b",
        r"\bfertilizer(s)?\b",
        r"\bpesticide(s)?\b",
        r"\bplant disease(s)?\b|\bdisease\b|\bpest(s)?\b",
        r"\byield\b|\bproductivity\b",
        r"\bagronomy\b",
        r"\bprecision agriculture\b|\bsmart farming\b",
        r"\blivestock\b",
        r"\bsustainab(le|ility)\b",
        r"\bremote sensing\b",
        r"\bagri(culture)?\s*(ai|ml|machine learning)\b",
        r"\bsupply chain\b",
        r"\brisk prediction\b",
        r"\bclimate\b.*\bcrop\b|\bcrop\b.*\bclimate\b",
    ]

    def __init__(
        self,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.ollama_model = ollama_model or os.getenv("GRAPH_RAG_MODEL", "qwen2.5:7b-q4")
        self.ollama_fallback_model = os.getenv(
            "GRAPH_RAG_FALLBACK_MODEL",
            os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b"),
        )
        self.ollama_timeout = int(os.getenv("GRAPH_RAG_OLLAMA_TIMEOUT", "300"))
        self.ollama_max_wait_seconds = int(os.getenv("GRAPH_RAG_OLLAMA_MAX_WAIT", "0"))
        self.ollama_connect_timeout = float(os.getenv("GRAPH_RAG_OLLAMA_CONNECT_TIMEOUT", "10"))
        self.ollama_retries = max(0, int(os.getenv("GRAPH_RAG_OLLAMA_RETRIES", "1")))
        self.enable_external_sources = os.getenv("GRAPH_RAG_ENABLE_EXTERNAL_SOURCES", "true").lower() in {
            "1", "true", "yes", "on"
        }
        self.external_top_k = int(os.getenv("GRAPH_RAG_EXTERNAL_TOP_K", "4"))
        self.external_max_chars = int(os.getenv("GRAPH_RAG_EXTERNAL_MAX_CHARS", "700"))

        self.kg_builder = AgroKGBuilder()
        self.kg_builder.build()

        self.intent_parser = IntentParser(self.kg_builder)
        self.query_engine = GraphQueryEngine(self.kg_builder)
        self.external_orchestrator = ExternalRetrievalOrchestrator()

    def run(self, user_query: str, use_llm: bool = True) -> Dict:
        is_agri_query = self._is_agriculture_query(user_query)
        parsed: ParsedIntent = self.intent_parser.parse(user_query)

        qctx = self.query_engine.query(
            crop_name=parsed.crop,
            pest_name=parsed.pest,
            disease_name=parsed.disease,
            climate_conditions=parsed.climate_conditions,
            soil_type=parsed.soil_type,
            pesticide_name=parsed.pesticide,
        )

        kg_context_text = self.query_engine.format_context_for_llm(qctx)
        has_local_kb_context = bool(qctx.pests_found or qctx.diseases_found or qctx.treatments)

        if not is_agri_query:
            external_meta = {
                "enabled": bool(self.enable_external_sources),
                "attempted": False,
                "agris_called": False,
                "agricola_called": False,
                "pubag_called": False,
                "cabi_called": False,
                "agecon_called": False,
                "asabe_called": False,
                "agris_results": 0,
                "agricola_results": 0,
                "pubag_results": 0,
                "cabi_results": 0,
                "agecon_results": 0,
                "asabe_results": 0,
                "total_results": 0,
                "context_used": False,
                "source_counts": {},
                "skipped_non_agriculture": True,
            }
            grounding = {
                "allow_generation": True,
                "message": "Non-agriculture query detected. AGRIS retrieval was skipped.",
                "conservative_mode": False,
                "metadata_limited": False,
            }

            if use_llm:
                answer = self._generate_general_response(user_query)
            else:
                answer = "This query appears outside agriculture. AGRIS retrieval was skipped."

            return {
                "query": user_query,
                "parsed_intent": asdict(parsed),
                "context": asdict(qctx),
                "kg_context_text": kg_context_text,
                "response": answer,
                "engine": {
                    "type": "graph_rag",
                    "llm_model": self.ollama_model if use_llm else None,
                    "llm_enabled": bool(use_llm),
                    "ollama_base_url": self.ollama_base_url,
                    "external_sources": external_meta,
                    "grounding": grounding,
                    "graph_stimulation": {
                        "external_evidence_docs": 0,
                        "external_evidence_injected": False,
                    },
                },
            }

        external_context_text, external_meta, grounding = self._build_external_context(
            user_query,
            parsed,
            has_local_kb_context=has_local_kb_context,
        )
        kg_context_text = self._inject_external_signals(kg_context_text, external_meta)

        if grounding.get("allow_generation") and grounding.get("message"):
            external_context_text = (
                f"GROUNDING NOTE: {grounding['message']}\n\n"
                + (external_context_text or "")
            ).strip()

        if use_llm and grounding.get("allow_generation", True):
            answer = self._generate_with_ollama(user_query, parsed, kg_context_text, external_context_text)
        elif use_llm and not grounding.get("allow_generation", True):
            answer = grounding.get("message") or "No reliable external evidence retrieved from configured sources."
        else:
            answer = self._fallback_response(parsed, kg_context_text)

        return {
            "query": user_query,
            "parsed_intent": asdict(parsed),
            "context": asdict(qctx),
            "kg_context_text": kg_context_text,
            "response": answer,
            "engine": {
                "type": "graph_rag",
                "llm_model": self.ollama_model if use_llm else None,
                "llm_enabled": bool(use_llm),
                "ollama_base_url": self.ollama_base_url,
                "external_sources": external_meta,
                "grounding": grounding,
                "graph_stimulation": {
                    "external_evidence_docs": int(external_meta.get("total_results", 0)),
                    "external_evidence_injected": bool(external_meta.get("context_used", False)),
                },
            },
        }

    def _inject_external_signals(self, kg_context_text: str, external_meta: Dict) -> str:
        if not external_meta.get("context_used"):
            return kg_context_text

        lines = [
            "EXTERNAL EVIDENCE SIGNALS:",
            f"- Primary docs: {external_meta.get('primary_results', 0)}",
            f"- Enrichment docs: {external_meta.get('enrichment_results', 0)}",
            f"- AGRIS docs: {external_meta.get('agris_results', 0)}",
            f"- FAOSTAT docs: {external_meta.get('faostat_results', 0)}",
            f"- CGIAR docs: {external_meta.get('cgiar_results', 0)}",
            f"- ClimateData docs: {external_meta.get('climate_results', 0)}",
            f"- SoilData docs: {external_meta.get('soil_results', 0)}",
            f"- AGRICOLA docs: {external_meta.get('agricola_results', 0)}",
            f"- PubAg docs: {external_meta.get('pubag_results', 0)}",
            f"- CABI docs: {external_meta.get('cabi_results', 0)}",
            f"- AgEcon docs: {external_meta.get('agecon_results', 0)}",
            f"- ASABE docs: {external_meta.get('asabe_results', 0)}",
        ]

        source_group_counts = external_meta.get("source_group_counts") or {}
        if source_group_counts:
            group_bits = [f"{k}={v}" for k, v in sorted(source_group_counts.items())]
            lines.append("- Source groups: " + ", ".join(group_bits))

        base = (kg_context_text or "").strip()
        if not base:
            return "\n".join(lines)
        return f"{base}\n\n" + "\n".join(lines)

    def _generate_with_ollama(
        self,
        user_query: str,
        parsed: ParsedIntent,
        kg_context_text: str,
        external_context_text: str,
    ) -> str:
        prompt = self._build_prompt(user_query, parsed, kg_context_text, external_context_text)
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "20m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": 520,
            },
        }

        url = f"{self.ollama_base_url.rstrip('/')}/api/generate"
        try:
            r = self._post_with_retry(url, payload, max_wait_seconds=self.ollama_max_wait_seconds)
            r.raise_for_status()
            data = r.json()
            answer = (data.get("response") or "").strip()
            if answer:
                return self._finalize_answer(
                    answer,
                    url,
                    user_query,
                    parsed,
                    kg_context_text,
                    external_context_text,
                )
            return self._fallback_response(parsed, kg_context_text)
        except requests.exceptions.ReadTimeout as exc:
            fallback_answer = self._try_fallback_model(
                url,
                user_query,
                parsed,
                kg_context_text,
                external_context_text,
            )
            if fallback_answer:
                return self._finalize_answer(
                    fallback_answer,
                    url,
                    user_query,
                    parsed,
                    kg_context_text,
                    external_context_text,
                )
            return (
                "Graph context is available, but local LLM generation timed out. "
                f"Model '{self.ollama_model}' exceeded {self.ollama_max_wait_seconds}s.\n\n"
                + self._fallback_response(parsed, kg_context_text)
            )
        except Exception as exc:
            return (
                "Graph context is available, but local LLM generation failed. "
                f"Reason: {exc}.\n\n"
                + self._fallback_response(parsed, kg_context_text)
            )

    def _finalize_answer(
        self,
        answer: str,
        url: str,
        user_query: str,
        parsed: ParsedIntent,
        kg_context_text: str,
        external_context_text: str,
    ) -> str:
        candidate = (answer or "").strip()
        if not candidate:
            return self._fallback_response(parsed, kg_context_text)

        if self._is_incomplete_response(candidate):
            completed = self._retry_for_complete_response(
                url,
                user_query,
                parsed,
                kg_context_text,
                external_context_text,
            )
            if completed:
                candidate = completed

        return self._append_minimum_completion(candidate)

    def _post_with_retry(self, url: str, payload: Dict, max_wait_seconds: Optional[int] = None) -> requests.Response:
        last_exc: Optional[Exception] = None
        budget = int(max_wait_seconds or self.ollama_max_wait_seconds)
        unlimited_wait = budget <= 0
        start = time.monotonic()

        for attempt in range(self.ollama_retries + 1):
            if unlimited_wait:
                remaining_budget = None
            else:
                elapsed = time.monotonic() - start
                remaining_budget = budget - elapsed
                if remaining_budget <= 0:
                    break

            req_payload = payload
            if attempt > 0:
                req_payload = json.loads(json.dumps(payload))
                options = req_payload.setdefault("options", {})
                base_predict = int(options.get("num_predict", 512))
                options["num_predict"] = max(180, int(base_predict * (0.8 ** attempt)))

            try:
                if unlimited_wait:
                    request_timeout = (self.ollama_connect_timeout, None)
                else:
                    request_timeout = (
                        self.ollama_connect_timeout,
                        min(self.ollama_timeout, max(5.0, remaining_budget)),
                    )

                return requests.post(
                    url,
                    json=req_payload,
                    timeout=request_timeout,
                )
            except requests.exceptions.ReadTimeout as exc:
                last_exc = exc
                continue
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                break

        if last_exc is not None:
            raise last_exc
        if unlimited_wait:
            raise requests.exceptions.ReadTimeout("Ollama request exceeded retry policy")
        raise requests.exceptions.ReadTimeout(
            f"Timed out after {budget}s waiting for Ollama response"
        )

    def _try_fallback_model(
        self,
        url: str,
        user_query: str,
        parsed: ParsedIntent,
        kg_context_text: str,
        external_context_text: str,
    ) -> Optional[str]:
        fallback_model = (self.ollama_fallback_model or "").strip()
        if not fallback_model or fallback_model == self.ollama_model:
            return None

        fallback_prompt = self._build_prompt(user_query, parsed, kg_context_text, external_context_text)
        fallback_payload = {
            "model": fallback_model,
            "prompt": fallback_prompt,
            "stream": False,
            "keep_alive": "20m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 3072,
                "num_predict": 340,
            },
        }

        try:
            r = self._post_with_retry(
                url,
                fallback_payload,
                max_wait_seconds=min(45, self.ollama_max_wait_seconds),
            )
            r.raise_for_status()
            data = r.json()
            answer = (data.get("response") or "").strip()
            if answer:
                return answer
        except Exception:
            return None
        return None

    def _build_prompt(
        self,
        user_query: str,
        parsed: ParsedIntent,
        kg_context_text: str,
        external_context_text: str,
    ) -> str:
        safety_lines = [
            "You are an intelligent Agriculture Research Assistant.",
            "Primary evidence source is AGRIS (FAO agricultural research database).",
            "Use retrieval-augmented generation and graph-based reasoning for agriculture queries.",
            "If query is not agriculture-related, answer normally and do NOT rely on AGRIS retrieval.",
            "For agriculture queries, prioritize AGRIS evidence; use complementary sources only for graph enrichment and validation.",
            "Do not hallucinate papers, datasets, or results. If evidence is limited, state reduced confidence clearly.",
            "Prefer recent evidence (after 2015), strong relevance, and cross-source consistency.",
            "Focus on practical, scientifically consistent, and actionable guidance.",
            "Avoid raw citations unless explicitly requested.",
        ]

        parsed_block = json.dumps(asdict(parsed), ensure_ascii=True, indent=2)

        return (
            "\n".join(safety_lines)
            + "\n\n"
            + f"USER QUERY:\n{user_query}\n\n"
            + f"PARSED INTENT:\n{parsed_block}\n\n"
            + "GRAPH CONTEXT:\n"
            + (kg_context_text or "(no graph context found)")
            + "\n\n"
            + "EXTERNAL RESEARCH CONTEXT (AGRIS primary + enrichment datasets/sources):\n"
            + (external_context_text or "(no external context used)")
            + "\n\n"
            + "Generate final answer with this structure:\n"
            + "1) Concept explanation\n"
            + "2) Research-backed insights (AGRIS-first, then supporting sources)\n"
            + "3) Graph-derived insights (key relationships and why they matter)\n"
            + "4) Practical recommendations (ordered, measurable if possible)\n"
            + "5) Optional comparisons (methods/models/inputs) when relevant\n"
            + "6) Confidence note (especially when AGRIS evidence is limited)\n"
        )

    def _build_general_prompt(self, user_query: str) -> str:
        return (
            "You are a helpful assistant. Answer the user query clearly and accurately. "
            "Use concise headings and practical points when helpful.\n\n"
            f"USER QUERY:\n{user_query}"
        )

    def _generate_general_response(self, user_query: str) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": self._build_general_prompt(user_query),
            "stream": False,
            "keep_alive": "20m",
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 2048,
                "num_predict": 320,
            },
        }

        url = f"{self.ollama_base_url.rstrip('/')}/api/generate"
        try:
            r = self._post_with_retry(url, payload, max_wait_seconds=self.ollama_max_wait_seconds)
            r.raise_for_status()
            data = r.json()
            answer = (data.get("response") or "").strip()
            if answer:
                return answer
        except Exception:
            pass

        return "I can help with that. Please share a bit more detail so I can give a precise answer."

    def _is_agriculture_query(self, user_query: str) -> bool:
        text = (user_query or "").lower()
        if not text.strip():
            return False

        for pattern in self._AGRI_DOMAIN_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _build_external_context(self, user_query: str, parsed: ParsedIntent, has_local_kb_context: bool):
        default_meta = {
            "enabled": bool(self.enable_external_sources),
            "attempted": False,
            "agris_called": False,
            "faostat_called": False,
            "cgiar_called": False,
            "climate_called": False,
            "soil_called": False,
            "agricola_called": False,
            "pubag_called": False,
            "cabi_called": False,
            "agecon_called": False,
            "asabe_called": False,
            "agris_results": 0,
            "faostat_results": 0,
            "cgiar_results": 0,
            "climate_results": 0,
            "soil_results": 0,
            "agricola_results": 0,
            "pubag_results": 0,
            "cabi_results": 0,
            "agecon_results": 0,
            "asabe_results": 0,
            "total_results": 0,
            "primary_results": 0,
            "enrichment_results": 0,
            "context_used": False,
            "source_counts": {},
            "source_group_counts": {},
        }
        default_grounding = {
            "allow_generation": True,
            "message": "",
            "conservative_mode": False,
            "metadata_limited": False,
        }

        if not self.enable_external_sources:
            return "", default_meta, default_grounding

        try:
            output = self.external_orchestrator.run(
                user_query=user_query,
                parsed_intent=parsed,
                has_local_kb_context=has_local_kb_context,
            )

            source_counts = output.retrieval.source_counts
            selected_docs = output.retrieval.documents
            source_group_counts = {}
            selected_source_counts = {}
            primary_results = 0
            enrichment_results = 0
            for doc in selected_docs:
                group = getattr(doc, "source_group", "research") or "research"
                source_group_counts[group] = source_group_counts.get(group, 0) + 1
                source_name = getattr(doc, "source", "") or "unknown"
                selected_source_counts[source_name] = selected_source_counts.get(source_name, 0) + 1
                if getattr(doc, "enrichment_only", False):
                    enrichment_results += 1
                else:
                    primary_results += 1

            def _count(source_name: str) -> int:
                return int(source_counts.get(source_name, 0) or 0)

            meta = {
                "enabled": True,
                "attempted": True,
                "agris_called": _count("AGRIS") > 0,
                "faostat_called": _count("FAOSTAT") > 0,
                "cgiar_called": _count("CGIAR") > 0,
                "climate_called": _count("ClimateData") > 0,
                "soil_called": _count("SoilData") > 0,
                "agricola_called": _count("AGRICOLA") > 0,
                "pubag_called": _count("PubAg") > 0,
                "cabi_called": _count("CABI") > 0,
                "agecon_called": _count("AgEcon") > 0,
                "asabe_called": _count("ASABE") > 0,
                "agris_results": _count("AGRIS"),
                "faostat_results": _count("FAOSTAT"),
                "cgiar_results": _count("CGIAR"),
                "climate_results": _count("ClimateData"),
                "soil_results": _count("SoilData"),
                "agricola_results": _count("AGRICOLA"),
                "pubag_results": _count("PubAg"),
                "cabi_results": _count("CABI"),
                "agecon_results": _count("AgEcon"),
                "asabe_results": _count("ASABE"),
                "total_results": output.retrieval.total_docs,
                "primary_results": primary_results,
                "enrichment_results": enrichment_results,
                "context_used": bool(output.context_text),
                "source_counts": source_counts,
                "selected_source_counts": selected_source_counts,
                "source_group_counts": source_group_counts,
                "source_call_logs": [x.to_dict() for x in output.retrieval.source_logs],
            }

            grounding = {
                "allow_generation": output.grounding.allow_generation,
                "message": output.grounding.message,
                "conservative_mode": output.grounding.conservative_mode,
                "metadata_limited": output.grounding.metadata_limited,
            }

            return output.context_text, meta, grounding
        except Exception:
            return "", default_meta, {
                "allow_generation": has_local_kb_context,
                "message": "No reliable external evidence retrieved from configured sources.",
                "conservative_mode": True,
                "metadata_limited": True,
            }

    def _should_use_external_context(self, parsed: ParsedIntent, user_query: str) -> bool:
        if not self.enable_external_sources:
            return False

        if not parsed.crop:
            return False

        if parsed.disease:
            return True

        disease_terms = {
            "disease", "blight", "blast", "mildew", "rust", "wilt",
            "spot", "rot", "fungal", "bacterial", "viral", "leaf",
        }
        query_tokens = set(re.findall(r"[a-zA-Z]+", (user_query or "").lower()))
        return bool(disease_terms & query_tokens)

    def _derive_disease_hint(self, user_query: str) -> str:
        text = (user_query or "").lower()
        for key in [
            "bacterial blight",
            "sheath blight",
            "blast",
            "powdery mildew",
            "leaf spot",
            "rust",
            "wilt",
            "rot",
        ]:
            if key in text:
                return key
        return "disease management"

    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    def _format_external_context(
        self,
        agris_results,
        agricola_results,
        pubag_results,
        cabi_results,
        agecon_results,
        asabe_results,
    ) -> str:
        if (
            not agris_results
            and not agricola_results
            and not pubag_results
            and not cabi_results
            and not agecon_results
            and not asabe_results
        ):
            return ""

        lines = []
        sources = [
            ("AGRIS", agris_results),
            ("AGRICOLA", agricola_results),
            ("PubAg", pubag_results),
            ("CABI", cabi_results),
            ("AgEcon", agecon_results),
            ("ASABE", asabe_results),
        ]

        for source_name, source_results in sources:
            if not source_results:
                continue
            lines.append(f"{source_name}:")
            for text in source_results[: self.external_top_k]:
                snippet = str(text).strip().replace("\n", " ")[: self.external_max_chars]
                if snippet:
                    lines.append(f"- {snippet}")

        return "\n".join(lines).strip()

    def _retry_for_complete_response(
        self,
        url: str,
        user_query: str,
        parsed: ParsedIntent,
        kg_context_text: str,
        external_context_text: str,
    ) -> Optional[str]:
        retry_prompt = (
            self._build_prompt(user_query, parsed, kg_context_text, external_context_text)
            + "\nIMPORTANT: Provide a complete final answer with all six sections and no unfinished bullets."
        )
        retry_payload = {
            "model": self.ollama_model,
            "prompt": retry_prompt,
            "stream": False,
            "keep_alive": "20m",
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": 760,
            },
        }

        try:
            r = self._post_with_retry(
                url,
                retry_payload,
                max_wait_seconds=min(50, self.ollama_max_wait_seconds),
            )
            r.raise_for_status()
            data = r.json()
            candidate = (data.get("response") or "").strip()
            if candidate and not self._is_incomplete_response(candidate):
                return candidate
        except Exception:
            return None
        return None

    def _is_incomplete_response(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True

        required_sections = [
            "Concept explanation",
            "Research-backed insights",
            "Graph-derived insights",
            "Practical recommendations",
            "Confidence note",
        ]
        missing_sections = [s for s in required_sections if s.lower() not in stripped.lower()]
        if missing_sections:
            return True

        if stripped.endswith(":"):
            return True

        last_line = stripped.splitlines()[-1].strip()
        if re.match(r"^[-*]\s*(\*\*[^*]+\*\*\s*:)?\s*$", last_line):
            return True

        if re.search(r"\b(apply|use|spray|treat)\s*$", last_line, flags=re.IGNORECASE):
            return True

        return False

    def _append_minimum_completion(self, text: str) -> str:
        cleaned = text.rstrip()

        cleaned = re.sub(
            r"\n+#{0,3}\s*5\)?\s*Optional comparisons\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\n+#{0,3}\s*6\)?\s*Confidence note\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        if cleaned.endswith(":"):
            cleaned += " Follow local label guidance with crop-specific products and PHI restrictions."

        if re.search(r"\b(and|or|of|to|for|with|in|on|at|from|by|during|under)\s*$", cleaned, flags=re.IGNORECASE):
            cleaned += " all product labels and local advisories."

        last_line = cleaned.splitlines()[-1].strip() if cleaned else ""
        if not re.search(r"[.!?]$", cleaned) and not re.match(r"^(#{1,6}|\d+\))", last_line):
            cleaned += "."

        cleaned = re.sub(
            r"\bAvoid\.\s*$",
            "Avoid incompatible tank mixes unless compatibility is confirmed.",
            cleaned,
            flags=re.IGNORECASE,
        )

        if "Confidence note".lower() not in cleaned.lower():
            cleaned += (
                "\n\n6) Confidence note\n"
                "- Confidence is moderate unless AGRIS and enrichment sources converge on the same recommendation."
            )

        return cleaned

    def _fallback_response(self, parsed: ParsedIntent, kg_context_text: str) -> str:
        lines = []
        lines.append("Situation summary:")
        lines.append(f"- Detected intent: {parsed.intent_type}")

        if parsed.crop:
            crop_name = self.kg_builder.G.nodes[parsed.crop].get("name_en", parsed.crop)
            lines.append(f"- Crop: {crop_name}")

        if parsed.climate_conditions:
            lines.append(f"- Climate flags: {', '.join(parsed.climate_conditions)}")

        lines.append("")
        lines.append("Actionable graph findings:")
        if kg_context_text.strip():
            lines.append(kg_context_text)
        else:
            lines.append("- No strong graph matches found for this query.")
            lines.append("- Suggestion: specify crop + visible symptom + weather condition.")

        lines.append("")
        lines.append("Safety notes:")
        lines.append("- Follow product label dose, PHI, and local agriculture advisories.")
        lines.append("- Avoid tank mixes unless compatibility is explicitly confirmed.")

        return "\n".join(lines)
