import asyncio
import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional

import networkx as nx

from .external_sources import fetch_agris, fetch_agricola
from .kg_data import (
    CLIMATE_CONDITIONS,
    CROPS,
    DISEASES,
    PESTICIDES,
    PESTS,
    RELATIONSHIPS,
    SOIL_TYPES,
)

_enrich_logger = logging.getLogger(__name__ + ".enrich")


class AgroKGBuilder:
    """Build and persist the TerraMind agricultural knowledge graph."""

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.node_index = {}

    def build(self) -> nx.MultiDiGraph:
        self._add_crop_nodes()
        self._add_pest_nodes()
        self._add_disease_nodes()
        self._add_pesticide_nodes()
        self._add_soil_nodes()
        self._add_climate_nodes()
        self._add_all_edges()
        self._validate()
        return self.G

    def _add_crop_nodes(self):
        for crop_id, data in CROPS.items():
            self.G.add_node(crop_id, node_type="crop", **data)
            self.node_index[data["name_en"].lower()] = crop_id
            self.node_index[data["name_hi"]] = crop_id
            self.node_index[crop_id] = crop_id
            if "scientific_name" in data:
                self.node_index[data["scientific_name"].lower()] = crop_id

    def _add_pest_nodes(self):
        for pest_id, data in PESTS.items():
            self.G.add_node(pest_id, node_type="pest", **data)
            self.node_index[data["name_en"].lower()] = pest_id
            self.node_index[data["name_hi"]] = pest_id
            self.node_index[pest_id] = pest_id
            self.node_index[data["scientific_name"].lower()] = pest_id

    def _add_disease_nodes(self):
        for disease_id, data in DISEASES.items():
            self.G.add_node(disease_id, node_type="disease", **data)
            self.node_index[data["name_en"].lower()] = disease_id
            self.node_index[data["name_hi"]] = disease_id
            self.node_index[disease_id] = disease_id

    def _add_pesticide_nodes(self):
        for pesticide_id, data in PESTICIDES.items():
            self.G.add_node(pesticide_id, node_type="pesticide", **data)
            self.node_index[data["name_en"].lower()] = pesticide_id
            self.node_index[data["name_hi"]] = pesticide_id
            self.node_index[pesticide_id] = pesticide_id

    def _add_soil_nodes(self):
        for soil_id, data in SOIL_TYPES.items():
            self.G.add_node(soil_id, node_type="soil_type", **data)
            self.node_index[data["name_en"].lower()] = soil_id
            self.node_index[data["name_hi"]] = soil_id
            self.node_index[soil_id] = soil_id

    def _add_climate_nodes(self):
        for climate_id, data in CLIMATE_CONDITIONS.items():
            self.G.add_node(climate_id, node_type="climate", **data)
            self.node_index[data["name_en"].lower()] = climate_id
            self.node_index[climate_id] = climate_id

    def _add_all_edges(self):
        rels = RELATIONSHIPS

        for r in rels["crop_pest"]:
            self.G.add_edge(
                r["crop"],
                r["pest"],
                relation="SUSCEPTIBLE_TO",
                **{k: v for k, v in r.items() if k not in ["crop", "pest"]},
            )

        for r in rels["crop_disease"]:
            self.G.add_edge(
                r["crop"],
                r["disease"],
                relation="VULNERABLE_TO",
                **{k: v for k, v in r.items() if k not in ["crop", "disease"]},
            )

        for r in rels["disease_treatment"]:
            self.G.add_edge(
                r["disease"],
                r["pesticide"],
                relation="TREATED_BY",
                **{k: v for k, v in r.items() if k not in ["disease", "pesticide"]},
            )

        for r in rels["pest_treatment"]:
            self.G.add_edge(
                r["pest"],
                r["pesticide"],
                relation="CONTROLLED_BY",
                **{k: v for k, v in r.items() if k not in ["pest", "pesticide"]},
            )

        for r in rels["pesticide_soil_conflict"]:
            self.G.add_edge(
                r["pesticide"],
                r["soil"],
                relation="CONFLICTS_WITH",
                **{k: v for k, v in r.items() if k not in ["pesticide", "soil"]},
            )

        for r in rels["pest_climate"]:
            self.G.add_edge(
                r["pest"],
                r["climate"],
                relation="PEAKS_DURING",
                **{k: v for k, v in r.items() if k not in ["pest", "climate"]},
            )

        for r in rels["tank_mix_conflicts"]:
            for a, b in [(r["pesticide_a"], r["pesticide_b"]), (r["pesticide_b"], r["pesticide_a"])]:
                self.G.add_edge(
                    a,
                    b,
                    relation="INCOMPATIBLE_WITH",
                    reason=r["reason"],
                    severity=r["severity"],
                )

        for r in rels["disease_climate"]:
            self.G.add_edge(
                r["disease"],
                r["climate"],
                relation="FAVORED_BY",
                **{k: v for k, v in r.items() if k not in ["disease", "climate"]},
            )

    def _validate(self):
        import logging

        log = logging.getLogger("AgroKGBuilder")

        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == "disease":
                treatments = [
                    v
                    for _, v, d in self.G.out_edges(node, data=True)
                    if d.get("relation") == "TREATED_BY"
                ]
                if not treatments:
                    log.warning("Disease with no treatment: %s", node)

        log.info(
            "KG built: %s nodes, %s edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )

    def resolve_node(self, name: str) -> Optional[str]:
        if not name:
            return None
        if name in self.node_index:
            return self.node_index[name]
        if name.lower() in self.node_index:
            return self.node_index[name.lower()]

        keys = list(self.node_index.keys())

        try:
            from rapidfuzz import fuzz, process

            match = process.extractOne(
                name.lower(),
                keys,
                scorer=fuzz.WRatio,
                score_cutoff=80,
            )
            if match:
                return self.node_index[match[0]]
        except Exception:
            from difflib import SequenceMatcher

            best_key = None
            best_score = 0.0
            target = name.lower()
            for key in keys:
                score = SequenceMatcher(None, target, key).ratio()
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key and best_score >= 0.8:
                return self.node_index[best_key]
        return None

    def save(self, path: str = "graph_rag/agrokg.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"graph": self.G, "node_index": self.node_index}, f)

    @classmethod
    def load(cls, path: str = "graph_rag/agrokg.pkl"):
        obj = cls()
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            obj.G = data["graph"]
            obj.node_index = data["node_index"]
        except FileNotFoundError:
            obj.build()
            obj.save(path)
        return obj


# ── Module-level enrichment function ─────────────────────────────────────────

# Shared builder instance used by the enrichment function.
_shared_builder: Optional[AgroKGBuilder] = None


def _get_shared_builder() -> AgroKGBuilder:
    """Lazily initialize and return the shared AgroKGBuilder instance."""
    global _shared_builder
    if _shared_builder is None:
        _shared_builder = AgroKGBuilder.load()
    return _shared_builder


async def enrich_graph_for_disease(crop: str, disease: str) -> List[str]:
    """
    Enrich the knowledge graph with external research data for a given
    crop-disease pair.  Calls both AGRIS and AGRICOLA fetchers, then
    upserts the returned document strings as new nodes in the graph.

    Parameters
    ----------
    crop : str
        The crop name (e.g. ``"Tomato"``).
    disease : str
        The disease label (e.g. ``"Late_blight"``).

    Returns
    -------
    list[str]
        Combined list of all retrieved text strings (may be empty).
    """
    _enrich_logger.info("Enriching graph for crop=%s, disease=%s", crop, disease)

    agris_results, agricola_results = await asyncio.gather(
        fetch_agris(crop, disease),
        fetch_agricola(crop, disease),
    )

    all_texts: List[str] = agris_results + agricola_results

    if not all_texts:
        _enrich_logger.warning(
            "Both AGRIS and AGRICOLA returned empty results for "
            "crop='%s', disease='%s'. Report will rely solely on "
            "local Knowledge Graph data.",
            crop, disease,
        )
        return all_texts

    # Upsert enrichment nodes into the shared graph
    builder = _get_shared_builder()
    disease_clean = disease.replace("___", " ").replace("__", " ").replace("_", " ")

    for idx, text in enumerate(all_texts):
        source = "agris" if idx < len(agris_results) else "agricola"
        node_id = re.sub(r"[^a-zA-Z0-9]", "_", f"enrich_{crop}_{disease}_{source}_{idx}").lower()

        if node_id not in builder.G.nodes:
            builder.G.add_node(
                node_id,
                node_type="enrichment",
                source=source,
                crop=crop,
                disease=disease_clean,
                text=text[:500],
                name_en=f"{crop} {disease_clean} ({source} #{idx + 1})",
            )
            builder.node_index[node_id] = node_id
            _enrich_logger.debug("Added enrichment node: %s", node_id)

    _enrich_logger.info(
        "Graph enrichment complete: %d new texts added (%d AGRIS, %d AGRICOLA)",
        len(all_texts), len(agris_results), len(agricola_results),
    )
    return all_texts

