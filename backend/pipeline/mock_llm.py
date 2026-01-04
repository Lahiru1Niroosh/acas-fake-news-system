import hashlib
import re
from typing import Dict, List

KEYWORD_POSITIVE = re.compile(r"\b(study|peer-reviewed|research|journal|official|cdc|who)\b", re.I)
KEYWORD_NEGATIVE = re.compile(r"\b(miracle|cure|conspiracy|hoax|fake|unproven|claim)\b", re.I)


def _deterministic_base_score(text: str) -> float:
    """Return a deterministic pseudo-random base score in [0,1) from text."""
    if not text:
        return 0.5
    h = hashlib.md5(text.encode("utf-8")).digest()
    val = int.from_bytes(h, "big") % 1000
    return val / 1000.0


class MockLLM:
    """A tiny, deterministic LLM-like simulator used for local development.

    It produces repeatable "analyses" and a synthesis from multiple agent outputs.
    """

    def analyze(self, text: str, image_url: str | None = None) -> Dict:
        base = _deterministic_base_score(text or "")
        plus = 0.0
        minus = 0.0
        evidence: List[str] = []

        if KEYWORD_POSITIVE.search(text or ""):
            plus += 0.18
            evidence.append("Contains references to studies or official sources")

        if KEYWORD_NEGATIVE.search(text or ""):
            minus += 0.22
            evidence.append("Contains sensational or unverified claims")

        if image_url:
            # Slight boost if an image is present (simulating corroborating visual evidence)
            plus += 0.06
            evidence.append("Image provided - visual evidence considered")

        score = max(0.0, min(1.0, base + plus - minus))
        label = "REAL" if score >= 0.5 else "FAKE"
        reason = (
            f"Deterministic base={base:.2f}, adj={(plus-minus):+.2f}. "
            f"Heuristics matched: {', '.join(evidence) if evidence else 'none'}"
        )

        return {
            "label": label,
            "score": round(score, 3),
            "reason": reason,
            "evidence": evidence,
        }

    def compare_similarity(self, text: str, existing_corpus: List[str] | None = None) -> Dict:
        """Return a deterministic similarity score vs a corpus (simulated)."""
        if not text:
            return {"similarity": 0.0, "matches": []}
        # Use base score on the text and treat it as similarity proxy
        sim = _deterministic_base_score(text)
        matches = []
        if existing_corpus:
            for c in existing_corpus:
                if c and c.lower() in text.lower():
                    matches.append(c)
        return {"similarity": round(sim, 3), "matches": matches}

    def synthesize(self, analyses: List[Dict]) -> Dict:
        """Combine several analyses into a final verdict and explanation."""
        if not analyses:
            return {"final_label": "REAL", "score": 0.5, "reason": "No analyses provided"}
        total = sum(a.get("score", 0.5) for a in analyses)
        avg = total / len(analyses)
        label = "REAL" if avg >= 0.5 else "FAKE"
        reasons = [a.get("reason", "") for a in analyses]
        reason = " | ".join(reasons)
        return {"final_label": label, "score": round(avg, 3), "reason": reason, "analyses": analyses}
