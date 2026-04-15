import logging
from typing import Optional

from app.chatbot.router.types import RetrieverContext
from graph_rag.graph_builder import AgroKGBuilder
from graph_rag.query_engine import GraphQueryEngine
from graph_rag.intent_parser import IntentParser

logger = logging.getLogger(__name__)

class GraphRetriever:
    """Retriever wrapper for the NetworkX AgroKG graph."""
    
    def __init__(self):
        try:
            self.kb = AgroKGBuilder.load()
            self.query_engine = GraphQueryEngine(self.kb)
            self.intent_parser = IntentParser(self.kb)
        except Exception as exc:
            logger.error(f"Failed to load AgroKG: {exc}")
            self.kb = None
            self.query_engine = None
            self.intent_parser = None

    async def retrieve(self, query: str, diag_env: Optional[dict] = None, top_k: Optional[int] = None) -> RetrieverContext:
        if not self.kb or not self.query_engine:
            return RetrieverContext(
                text_content="Graph RAG is not initialized.",
                metadata={"error": "kg_missing"}
            )
            
        diag_env = diag_env or {}
        identified_crop = diag_env.get("identified_crop")
        identified_class = diag_env.get("identified_class")
        
        # Merge diag env logic into query extraction implicitly via intent_parser
        parsed = self.intent_parser.parse(query)
        
        crop = parsed.crop or identified_crop
        disease = parsed.disease or (identified_class.split("___")[-1] if identified_class else None)
        pest = parsed.pest
        soil_type = parsed.soil_type
        pesticide = parsed.pesticide
        climate_conditions = parsed.climate_conditions
        
        qctx = self.query_engine.query(
            crop_name=crop,
            pest_name=pest,
            disease_name=disease,
            climate_conditions=climate_conditions,
            soil_type=soil_type,
            pesticide_name=pesticide,
        )
        
        # If standard parsing found nothing, fallback to pure string crop/disease search
        if not any([qctx.crop, qctx.pests_found, qctx.diseases_found, qctx.treatments, qctx.high_risk_pests_now]):
            if identified_crop or identified_class:
                logger.info("Graph RAG standard parse empty, using diag context fallback.")
                fallback_disease = identified_class.replace("___", " ").replace("__", " ").replace("_", " ") if identified_class else None
                qctx = self.query_engine.query(crop_name=identified_crop, disease_name=fallback_disease)

        kg_text = self.query_engine.format_context_for_llm(qctx)
        
        # Determine if graph found anything substantial
        nodes_found = len(qctx.treatments) + len(qctx.diseases_found) + len(qctx.pests_found) + len(qctx.soil_conflicts) + len(qctx.tank_mix_warnings)
        
        return RetrieverContext(
            text_content="KNOWLEDGE GRAPH DATA:\n" + kg_text if kg_text.strip() else "",
            metadata={
                "nodes_found": nodes_found,
                "parsed_intent": parsed.__dict__ if parsed else None,
                "sources": [{"file_name": "AgroKG v1.0", "page": "-", "snippet": kg_text[:200], "score": 1.0}] if kg_text.strip() else []
            }
        )
