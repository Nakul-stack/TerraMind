import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedIntent:
    intent_type: str
    crop: Optional[str] = None
    pest: Optional[str] = None
    disease: Optional[str] = None
    climate_conditions: List[str] = field(default_factory=list)
    soil_type: Optional[str] = None
    pesticide: Optional[str] = None
    language_hint: str = "en"


class IntentParser:
    """Simple rule-based parser for agricultural advisory questions."""

    def __init__(self, kg_builder):
        self.kg = kg_builder

        self.intent_patterns = {
            "crop_risk_assessment": [
                r"risk.*(?:for|in)\s+(\w+)",
                r"what.*(?:pest|disease).*(?:in|for)\s+(\w+)",
                r"alerts?.*(?:for|in)\s+(\w+)",
            ],
            "pest_management": [
                r"how.*control\s+(.+)",
                r"manage\s+(.+)",
                r"treatment.*(?:for)?\s+(.+)",
                r"spray.*(?:for)?\s+(.+)",
            ],
            "disease_management": [
                r"how.*treat\s+(.+)",
                r"disease.*(?:in|for)\s+(\w+)",
                r"fungicide.*(?:for)?\s+(.+)",
            ],
            "climate_alert": [
                r"(?:rain|humid|temperature|weather).*(?:risk|alert|issue)",
                r"current weather.*(?:what|which).*(?:pest|disease)",
                r"hot and humid",
                r"high humidity",
            ],
            "mix_safety": [
                r"mix\s+(.+)\s+(?:and|with)\s+(.+)",
                r"tank mix",
                r"compatible\s+with",
            ],
            "soil_compatibility": [
                r"soil.*(?:safe|compatib|suitable)",
                r"(?:alkaline|acidic|black|red|sandy|clay).*(?:soil)",
            ],
        }

        self.climate_keywords = {
            "high_humidity": ["high humidity", "humid", "moist"],
            "moderate_humidity": ["moderate humidity"],
            "low_humidity": ["low humidity", "dry air"],
            "high_temperature": ["high temp", "hot", "heat"],
            "moderate_temperature": ["moderate temp", "warm"],
            "low_temperature": ["low temp", "cold", "cool"],
            "heavy_rain": ["heavy rain", "raining", "rainy", "monsoon", "downpour"],
            "drizzle": ["drizzle", "light rain"],
            "drought": ["drought", "water stress", "very dry"],
            "windy": ["windy", "high wind", "strong wind"],
            "cloudy": ["cloudy", "overcast"],
        }

        self.soil_keywords = {
            "alkaline_soil": ["alkaline soil", "high ph soil", "basic soil"],
            "acidic_soil": ["acidic soil", "low ph soil", "sour soil"],
            "black_soil": ["black soil", "regur"],
            "red_soil": ["red soil"],
            "sandy_soil": ["sandy soil", "sand soil"],
            "clay_soil": ["clay soil", "heavy soil"],
            "loamy_soil": ["loamy soil", "loam"],
            "waterlogged": ["waterlogged", "water logging"],
        }

    def parse(self, user_query: str) -> ParsedIntent:
        q = (user_query or "").strip().lower()
        language_hint = self._detect_language(q)

        intent_type = self._classify_intent(q)
        crop = self._extract_crop(q)
        pest = self._extract_entity(q, node_type="pest")
        disease = self._extract_entity(q, node_type="disease")
        pesticides = self._extract_entities(q, node_type="pesticide")
        pesticide = None
        if pesticides:
            if intent_type == "mix_safety" and len(pesticides) >= 2:
                pesticide = ",".join(pesticides[:2])
            else:
                pesticide = pesticides[0]
        climate_conditions = self._extract_climate_conditions(q)
        soil_type = self._extract_soil_type(q)

        return ParsedIntent(
            intent_type=intent_type,
            crop=crop,
            pest=pest,
            disease=disease,
            climate_conditions=climate_conditions,
            soil_type=soil_type,
            pesticide=pesticide,
            language_hint=language_hint,
        )

    def _detect_language(self, q: str) -> str:
        devanagari_range = re.compile(r"[\u0900-\u097F]")
        return "hi" if devanagari_range.search(q) else "en"

    def _classify_intent(self, q: str) -> str:
        scores = {k: 0 for k in self.intent_patterns.keys()}

        for intent, patterns in self.intent_patterns.items():
            for pat in patterns:
                if re.search(pat, q):
                    scores[intent] += 2

        if any(k in q for k in ["weather", "rain", "humidity", "humid", "temperature", "hot", "cold"]):
            scores["climate_alert"] += 1
        if any(k in q for k in ["mix", "compatible", "tank"]):
            scores["mix_safety"] += 1
        if any(k in q for k in ["soil", "ph", "alkaline", "acidic", "sandy", "clay"]):
            scores["soil_compatibility"] += 1
        if any(k in q for k in ["pest", "insect", "worm", "borer", "aphid"]):
            scores["pest_management"] += 1
        if any(k in q for k in ["disease", "fungus", "blight", "rot", "mildew", "rust"]):
            scores["disease_management"] += 1

        best = max(scores.items(), key=lambda x: x[1])
        if best[1] == 0:
            return "general_advisory"

        if best[0] in ["pest_management", "disease_management"]:
            if any(k in q for k in ["risk", "alert", "forecast", "current"]):
                return "crop_risk_assessment"

        return best[0]

    def _extract_crop(self, q: str) -> Optional[str]:
        for node_id, attrs in self.kg.G.nodes(data=True):
            if attrs.get("node_type") != "crop":
                continue
            name_en = (attrs.get("name_en") or "").lower()
            name_hi = (attrs.get("name_hi") or "").lower()
            if name_en and name_en in q:
                return node_id
            if name_hi and name_hi in q:
                return node_id

        token_candidates = re.findall(r"\b[a-z][a-z_]{2,}\b", q)
        for token in token_candidates:
            node_id = self.kg.resolve_node(token)
            if node_id and self.kg.G.nodes[node_id].get("node_type") == "crop":
                return node_id
        return None

    def _extract_entity(self, q: str, node_type: str) -> Optional[str]:
        best_match = None
        best_len = 0

        for node_id, attrs in self.kg.G.nodes(data=True):
            if attrs.get("node_type") != node_type:
                continue
            names = [
                (attrs.get("name_en") or "").lower(),
                (attrs.get("name_hi") or "").lower(),
                (attrs.get("scientific_name") or "").lower(),
            ]
            for n in names:
                if n and n in q and len(n) > best_len:
                    best_match = node_id
                    best_len = len(n)

        if best_match:
            return best_match

        token_candidates = re.findall(r"\b[a-z][a-z_]{2,}\b", q)
        for token in token_candidates:
            node_id = self.kg.resolve_node(token)
            if node_id and self.kg.G.nodes[node_id].get("node_type") == node_type:
                return node_id

        return None

    def _extract_entities(self, q: str, node_type: str) -> List[str]:
        matches = []
        for node_id, attrs in self.kg.G.nodes(data=True):
            if attrs.get("node_type") != node_type:
                continue
            names = [
                (attrs.get("name_en") or "").lower(),
                (attrs.get("name_hi") or "").lower(),
                (attrs.get("scientific_name") or "").lower(),
            ]
            if any(n and n in q for n in names):
                matches.append(node_id)

        # Keep insertion order while deduplicating.
        deduped = []
        for m in matches:
            if m not in deduped:
                deduped.append(m)

        if deduped:
            return deduped

        # Fallback token-wise fuzzy resolution.
        token_candidates = re.findall(r"\b[a-z][a-z_]{2,}\b", q)
        for token in token_candidates:
            node_id = self.kg.resolve_node(token)
            if node_id and self.kg.G.nodes[node_id].get("node_type") == node_type and node_id not in deduped:
                deduped.append(node_id)
        return deduped

    def _extract_climate_conditions(self, q: str) -> List[str]:
        found = []
        for climate_id, keywords in self.climate_keywords.items():
            for kw in keywords:
                if kw in q:
                    found.append(climate_id)
                    break

        if "temperature" in q and not any(c.endswith("temperature") for c in found):
            nums = re.findall(r"(\d{2})\s*\D*c", q)
            if nums:
                try:
                    t = int(nums[0])
                    if t >= 33:
                        found.append("high_temperature")
                    elif t <= 18:
                        found.append("low_temperature")
                    else:
                        found.append("moderate_temperature")
                except Exception:
                    pass

        if "humidity" in q and not any(c.endswith("humidity") for c in found):
            nums = re.findall(r"(\d{2,3})\s*%", q)
            if nums:
                try:
                    h = int(nums[0])
                    if h >= 80:
                        found.append("high_humidity")
                    elif h <= 40:
                        found.append("low_humidity")
                    else:
                        found.append("moderate_humidity")
                except Exception:
                    pass

        uniq = []
        for c in found:
            if c not in uniq:
                uniq.append(c)
        return uniq

    def _extract_soil_type(self, q: str) -> Optional[str]:
        for soil_id, keywords in self.soil_keywords.items():
            for kw in keywords:
                if kw in q:
                    return soil_id

        node_id = self.kg.resolve_node(q)
        if node_id and self.kg.G.nodes[node_id].get("node_type") == "soil_type":
            return node_id

        return None
