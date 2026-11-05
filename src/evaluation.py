"""
Evaluación compartida por los Experimentos 1 y 2:
- Juez LLM ciego al sistema evaluado. Si no devuelve un veredicto válido, la fila queda
  sin veredicto y no cuenta como acierto ni como error.
- Intervalo de Wilson y prueba exacta de McNemar para el análisis estadístico.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.inference_setup import LocalInferenceEngine

ZEP_RAW_LABEL = "Graphiti (sin filtro)"
ZEP_ASOF_LABEL = "Graphiti + capa as-of"
LIGHTRAG_LABEL = "LightRAG (Estático)"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial judge of question-answering systems. Decide whether the "
    "System Response gives the Gold Answer to the Question. Accept paraphrases and "
    "equivalent values, and ignore extra detail that does not contradict the Gold Answer. "
    "If the response refuses, says it does not know, or gives another value, it is incorrect.\n"
    "If Outdated Values are listed, set 'anachronistic' to true when the response presents "
    "any of them as the answer, or lists them next to the Gold Answer without saying which "
    "one applies. Mentioning one explicitly as no longer applicable is not anachronistic.\n"
    'Respond ONLY with JSON: {"correct": true|false, "anachronistic": true|false, '
    '"rationale": "one short sentence"}'
)


class AnswerJudge:
    """Compara una respuesta con la respuesta oro sin saber qué sistema la generó."""

    def __init__(self, engine: LocalInferenceEngine) -> None:
        self.engine = engine

    async def evaluate(
        self,
        question: str,
        gold_answer: str,
        response: str,
        traps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        user_prompt = (
            f"Question: {question}\n"
            f"Gold Answer: {gold_answer}\n"
            f"Outdated Values: {' | '.join(traps) if traps else 'none'}\n"
            f"System Response: {response}\n\n"
            "JSON verdict:"
        )
        raw = await self.engine.generate_async(
            prompt=user_prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        verdict = self.parse_verdict(raw)
        if verdict is None:
            return {
                "correct": None,
                "anachronistic": None,
                "rationale": raw[:300],
                "source": "sin_veredicto",
            }
        if not traps:
            verdict["anachronistic"] = None
        return verdict

    @staticmethod
    def parse_verdict(raw: str) -> Optional[Dict[str, Any]]:
        """Extrae el veredicto solo si tiene los campos booleanos esperados."""
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data.get("correct"), bool) or not isinstance(
            data.get("anachronistic"), bool
        ):
            return None
        return {
            "correct": data["correct"],
            "anachronistic": data["anachronistic"],
            "rationale": str(data.get("rationale", "")),
            "source": "llm",
        }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Intervalo de confianza de Wilson al 95% para una proporción."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Valor p bilateral de McNemar exacto a partir de los pares discordantes."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm_adjust(p_values: List[float]) -> List[float]:
    """Ajuste de Holm-Bonferroni para las comparaciones planificadas de un conjunto de datos."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted
