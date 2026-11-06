"""
BLOQUE 3: Pipeline del Experimento 2 (Recuperación Bi-temporal: Zep/Graphiti vs LightRAG).
Las librerías son las reales e ingieren diálogo sin estructurar, así que cada sistema debe
extraer los hechos por sí mismo. Se comparan tres configuraciones:
- LightRAG, sin intervalos de validez.
- Graphiti sin filtro: entrega los hechos con sus metadatos temporales.
- Graphiti + capa as-of: una capa propia filtra esos metadatos según (VT, TT), porque
  Graphiti no ofrece consultas de ese tipo. Separarlas muestra qué aporta cada parte.
Casos evaluados:
- Preguntas 'temporal-reasoning' y 'knowledge-update' de LongMemEval, que evalúan razonamiento
  temporal general. Allí la capa as-of (VT = TT = fecha de la pregunta) es exploratoria.
- Escenarios sintéticos con hechos tardíos y correcciones retroactivas, donde VT y TT difieren:
  son los únicos que prueban la bi-temporalidad.
Las respuestas se evalúan con un Juez LLM ciego al sistema y se exportan a
'experimento2_resultados.csv', junto con los hechos usados para diagnosticar cada fallo.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.data_loader import (
    EvalCase,
    TemporalEpisode,
    load_bitemporal_evaluation_cases,
    load_longmemeval_temporal_cases,
)
from src.evaluation import AnswerJudge, LIGHTRAG_LABEL, ZEP_ASOF_LABEL, ZEP_RAW_LABEL
from src.exp1_indexing import ANSWER_SYSTEM_PROMPT, create_lightrag
from src.inference_setup import LLMGenerationError, LocalInferenceEngine

# Tantos grupos seguidos fallando indican un problema general, no casos difíciles:
# seguir solo gastaría cómputo.
MAX_CONSECUTIVE_FAILED_GROUPS = 10
# El valor por defecto de `OpenAIGenericClient`: con 4096 se cortaban extracciones legítimas
GRAPHITI_MAX_TOKENS = 16384


class _NoRetryOnTruncation:
    """
    Cuando Qwen entra en un bucle de repetición, el JSON queda cortado y Graphiti lo reintenta
    hasta 4 veces, pero con temperatura 0 cada intento repite el mismo bucle: en el piloto
    ningún reintento salvó un caso. Una salida cortada se reporta como error del caso sin
    reintentar, así que el resultado es el mismo con un cuarto del cómputo.
    """

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        response = await self._client.chat.completions.create(**kwargs)
        if response.choices and response.choices[0].finish_reason == "length":
            raise LLMGenerationError(
                f"Salida de Graphiti cortada en {kwargs.get('max_tokens')} tokens"
            )
        return response


# =============================================================================
# 1. Consulta as-of sobre los campos temporales de Graphiti
# =============================================================================


def _as_utc(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _iso_or_none(moment: Optional[datetime]) -> Optional[str]:
    value = _as_utc(moment)
    return value.isoformat() if value else None


def known_invalid_at(edge: Any, tt_clock: datetime) -> Optional[datetime]:
    """
    Devuelve el fin de validez que el sistema conocía en `tt_clock`. Graphiti sobrescribe
    `invalid_at` en la misma arista cuando aprende que un hecho terminó, y `expired_at`
    indica cuándo lo aprendió: antes de ese instante, ese fin de validez no se conocía.
    """
    invalid_at = _as_utc(edge.invalid_at)
    expired_at = _as_utc(edge.expired_at)
    if invalid_at is None or (expired_at is not None and expired_at > tt_clock):
        return None
    return invalid_at


def edge_visible_as_of(edge: Any, query_vt: datetime, tt_clock: datetime) -> bool:
    """
    Una arista es visible en (VT, TT) si ya estaba registrada en TT, ya era válida en VT y
    su validez no había terminado en VT según lo conocido en TT.
    Se aplica aquí porque `SearchFilters` compara campos por separado y no puede expresar
    que `invalid_at` solo cuenta si ya se conocía en TT.
    """
    if _as_utc(edge.created_at) > tt_clock:
        return False
    valid_at = _as_utc(edge.valid_at)
    if valid_at is not None and valid_at > query_vt:
        return False
    invalid_at = known_invalid_at(edge, tt_clock)
    return invalid_at is None or query_vt < invalid_at


# =============================================================================
# 2. Sistemas evaluados
# =============================================================================


class GraphitiMemory:
    """
    Memoria bi-temporal con `graphiti-core` sobre Neo4j, con un `group_id` por escenario.
    `created_at` y `expired_at` son el reloj real de ingesta, así que se guarda el instante
    en que terminó cada episodio para traducir un TT simulado a ese reloj.
    """

    # Varios grupos abren Graphiti a la vez y crean los mismos índices de Neo4j
    _index_lock = asyncio.Lock()

    def __init__(self, config: ExperimentConfig, engine: LocalInferenceEngine) -> None:
        self.config = config
        self.engine = engine
        self.graphiti: Any = None
        self.group_id: str = ""
        self.tt_clock: List[Tuple[datetime, datetime]] = []
        self.graph_edges: List[Dict[str, Any]] = []

    async def open(self, group_id: str) -> None:
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from graphiti_core.embedder.client import EmbedderClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        engine = self.engine

        class LocalEmbedder(EmbedderClient):
            async def create(self, input_data: Any) -> List[float]:
                text = input_data if isinstance(input_data, str) else input_data[0]
                return engine.encode_texts([text])[0].tolist()

            async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
                return engine.encode_texts(input_data_list).tolist()

        class NoOpReranker(CrossEncoderClient):
            # La búsqueda usa fusión RRF, que no necesita reordenador
            async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
                return [(passage, 1.0) for passage in passages]

        self.graphiti = Graphiti(
            self.config.neo4j_uri,
            self.config.neo4j_user,
            self.config.neo4j_password,
            llm_client=OpenAIGenericClient(
                config=LLMConfig(
                    api_key=self.config.api_key,
                    model=self.config.llm_model_id,
                    small_model=self.config.llm_model_id,
                    base_url=self.config.base_url,
                    temperature=0.0,
                    max_tokens=GRAPHITI_MAX_TOKENS,
                ),
                client=SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=_NoRetryOnTruncation(
                            AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
                        )
                    )
                ),
                max_tokens=GRAPHITI_MAX_TOKENS,
            ),
            embedder=LocalEmbedder(),
            cross_encoder=NoOpReranker(),
            max_coroutines=min(32, self.config.max_num_seqs),
        )
        async with GraphitiMemory._index_lock:
            await self.graphiti.build_indices_and_constraints()
        # Una corrida interrumpida pudo dejar datos de este mismo escenario
        self.group_id = re.sub(r"[^a-zA-Z0-9_-]", "_", group_id)
        await clear_data(self.graphiti.driver, group_ids=[self.group_id])
        self.tt_clock = []

    async def ingest(self, episodes: List[TemporalEpisode]) -> None:
        from graphiti_core.nodes import EpisodeType

        for episode in episodes:
            await self.graphiti.add_episode(
                name=episode.episode_id,
                episode_body=episode.text,
                source_description="conversation",
                reference_time=episode.recorded_at,
                source=EpisodeType.message,
                group_id=self.group_id,
            )
            self.tt_clock.append((episode.recorded_at, datetime.now(timezone.utc)))

        from graphiti_core.edges import EntityEdge

        edges = await EntityEdge.get_by_group_ids(self.graphiti.driver, [self.group_id])
        self.graph_edges = [
            {
                "fact": edge.fact,
                "valid_at": _iso_or_none(edge.valid_at),
                "invalid_at": _iso_or_none(edge.invalid_at),
                "expired_at": _iso_or_none(edge.expired_at),
            }
            for edge in edges
        ]

    def _clock_for(self, query_tt: datetime) -> Optional[datetime]:
        """Instante real en que el sistema tenía registrado lo ingerido hasta `query_tt`."""
        known = [clock for recorded_at, clock in self.tt_clock if recorded_at <= query_tt]
        return known[-1] if known else None

    async def answer(self, case: EvalCase, as_of: bool) -> Tuple[str, List[str]]:
        """Responde con hechos sin filtrar, o solo con los visibles en la ventana del caso."""
        top_k = self.config.exp2_top_k_retrieval
        edges = await self.graphiti.search(
            case.question,
            group_ids=[self.group_id],
            num_results=self.config.exp2_candidate_pool,
        )
        if as_of:
            clock = self._clock_for(case.query_tt)
            facts = []
            if clock is not None:
                visible = [e for e in edges if edge_visible_as_of(e, case.query_vt, clock)]
                facts = [
                    self._format_fact(e, known_invalid_at(e, clock)) for e in visible[:top_k]
                ]
        else:
            facts = [self._format_fact(e, _as_utc(e.invalid_at)) for e in edges[:top_k]]

        context = "\n".join(facts) if facts else "No facts are available."
        answer = await self.engine.generate_async(
            prompt=f"Facts:\n{context}\n\nQuestion: {case.question}\nConcise answer:",
            system_prompt=ANSWER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=300,
        )
        return answer, facts

    @staticmethod
    def _format_fact(edge: Any, invalid_at: Optional[datetime]) -> str:
        valid_at = _as_utc(edge.valid_at)
        start = valid_at.date().isoformat() if valid_at else "unknown"
        end = invalid_at.date().isoformat() if invalid_at else "present"
        return f"- {edge.fact} (valid from {start} until {end})"

    async def close(self) -> None:
        if self.graphiti is not None:
            from graphiti_core.utils.maintenance.graph_data_operations import clear_data

            await clear_data(self.graphiti.driver, group_ids=[self.group_id])
            await self.graphiti.close()
            self.graphiti = None


class LightRAGMemory:
    """LightRAG real: acumula el diálogo con sus fechas, sin intervalos de validez."""

    def __init__(self, config: ExperimentConfig, engine: LocalInferenceEngine) -> None:
        self.config = config
        self.engine = engine
        self.rag: Any = None
        self.work_dir: Any = None
        self.graph_edges: List[Dict[str, Any]] = []

    async def open(self, group_id: str) -> None:
        self.work_dir = self.config.base_dir / f"workdir_exp2_{group_id}"
        self.rag = await create_lightrag(self.config, self.engine, self.work_dir, group_id)

    async def ingest(self, episodes: List[TemporalEpisode]) -> None:
        failures_before = self.engine.llm_failure_count
        await self.rag.ainsert([episode.text for episode in episodes])
        self._raise_on_llm_failures(failures_before)

    async def answer(self, case: EvalCase, as_of: bool = False) -> Tuple[str, List[str]]:
        from lightrag import QueryParam

        failures_before = self.engine.llm_failure_count
        answer = await self.rag.aquery(
            case.question,
            param=QueryParam(mode="mix", enable_rerank=False, response_type="Single Paragraph"),
        )
        self._raise_on_llm_failures(failures_before)
        return str(answer), []

    def _raise_on_llm_failures(self, failures_before: int) -> None:
        # LightRAG captura los errores del LLM sin propagarlos
        if self.engine.llm_failure_count > failures_before:
            raise LLMGenerationError(
                f"LightRAG tuvo fallos del LLM: {self.engine.last_llm_error!r}"
            )

    async def close(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()
            self.rag = None
        if self.work_dir is not None:
            shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = None


# =============================================================================
# 3. Orquestador del Experimento 2 y Exportación a CSV
# =============================================================================


def _base_row(case: EvalCase, metodo: str) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "fuente_dataset": case.source,
        "escenario": case.group_id,
        "categoria_temporal": case.category,
        "metodo": metodo,
        "query_vt": case.query_vt.date().isoformat() if case.query_vt else "",
        "query_tt": case.query_tt.date().isoformat() if case.query_tt else "",
        "pregunta": case.question,
        "ground_truth": case.ground_truth,
        "trampas": " | ".join(case.anachronistic_traps),
    }


def _has_non_latin_letters(text: str) -> bool:
    """Detecta hechos en otro alfabeto: los episodios de los casos están en inglés."""
    return any(char.isalpha() and ord(char) > 0x24F for char in text)


def _diagnose(
    case: EvalCase, metodo: str, graph_edges: List[Dict[str, Any]], facts: List[str]
) -> Dict[str, Any]:
    """
    Registra en qué etapa estaba o se perdió el valor correcto, para separar errores de
    extracción, de fecha, de recuperación y de respuesta. Solo mide: no cambia la evaluación.
    LightRAG y LongMemEval no tienen valores comparables de esta forma.
    """
    if metodo == LIGHTRAG_LABEL:
        return {}
    graph_facts = [edge["fact"] for edge in graph_edges]
    language = {
        "proporcion_hechos_no_latinos": (
            sum(map(_has_non_latin_letters, graph_facts)) / len(graph_facts)
            if graph_facts
            else None
        )
    }
    if case.source != "sintetico":
        return language

    def _mentions(value: str, texts: List[str]) -> bool:
        return any(value.lower() in text.lower() for text in texts)

    expected_day = case.expected_valid_from.date().isoformat()
    return {
        **language,
        "oro_en_grafo": _mentions(case.ground_truth, graph_facts),
        "oro_en_hechos": _mentions(case.ground_truth, facts),
        "trampa_en_hechos": any(_mentions(trap, facts) for trap in case.anachronistic_traps),
        "fecha_oro_correcta": any(
            _mentions(case.ground_truth, [edge["fact"]])
            and (edge["valid_at"] or "").startswith(expected_day)
            for edge in graph_edges
        ),
        "aristas_grafo": json.dumps(graph_edges, ensure_ascii=False),
    }


async def _evaluate_group(
    system: Any,
    arms: Tuple[Tuple[str, bool], ...],
    judge: AnswerJudge,
    pending: List[Tuple[str, bool, EvalCase]],
) -> List[Dict[str, Any]]:
    """Indexa una vez los episodios del grupo y responde lo pendiente de cada configuración."""
    rows: List[Dict[str, Any]] = []
    try:
        await system.open(pending[0][2].group_id)
        await system.ingest(pending[0][2].episodes)
        for metodo, as_of, case in pending:
            answer, facts = await system.answer(case, as_of=as_of)
            verdict = await judge.evaluate(
                case.question, case.ground_truth, answer, case.anachronistic_traps
            )
            rows.append(
                {
                    **_base_row(case, metodo),
                    "estado": "ok",
                    "respuesta_generada": answer,
                    "hechos_usados": json.dumps(facts, ensure_ascii=False),
                    **_diagnose(case, metodo, system.graph_edges, facts),
                    "correcto": verdict["correct"],
                    "error_anacronico": verdict["anachronistic"],
                    "juez_fuente": verdict["source"],
                    "juez_rationale": verdict["rationale"],
                }
            )
    except Exception as exc:
        # Sin servidor, nada de lo que siga sería una medición válida
        if not system.engine.is_server_healthy():
            raise
        # Un error de un caso se reporta en la tasa de error del paper; la corrida continúa
        print(f"[ERROR] {pending[0][2].group_id}: {exc.__class__.__name__}: {exc}")
        answered = {(row["metodo"], row["case_id"]) for row in rows}
        for metodo, _, case in pending:
            if (metodo, case.case_id) not in answered:
                rows.append(
                    {
                        **_base_row(case, metodo),
                        "estado": "error",
                        "juez_fuente": "error",
                        "juez_rationale": f"{exc.__class__.__name__}: {exc}"[:500],
                    }
                )
    finally:
        await system.close()
    return rows


async def run_experiment_2(
    config: ExperimentConfig | None = None,
    engine: LocalInferenceEngine | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el Experimento 2 y exporta los resultados caso por caso a
    'experimento2_resultados.csv'. Retorna el detalle y un resumen por conjunto de datos.
    """
    cfg = config or ExperimentConfig()
    cfg.auto_configure_for_colab_gpu()
    inf_engine = engine or LocalInferenceEngine(cfg)
    if not inf_engine.is_server_healthy():
        print("[INFO] Servidor vLLM no detectado activo. Iniciando automáticamente...")
        inf_engine.start_vllm_server(timeout_seconds=900)

    cases = load_bitemporal_evaluation_cases(
        cfg.exp2_num_synthetic_scenarios, seed=cfg.exp2_scenario_seed
    ) + load_longmemeval_temporal_cases(cfg.exp2_max_longmemeval_cases, pilot=cfg.pilot)
    groups: Dict[str, List[EvalCase]] = {}
    for case in cases:
        groups.setdefault(case.group_id, []).append(case)

    # Las dos configuraciones de Graphiti reutilizan el mismo índice. Graphiti ingiere los
    # episodios de un grupo en serie y deja la GPU casi ociosa, así que varios grupos se
    # procesan a la vez. LightRAG va de a uno: detecta sus fallos con un contador global del
    # motor, que se mezclaría entre grupos simultáneos.
    systems = (
        (
            lambda: GraphitiMemory(cfg, inf_engine),
            ((ZEP_ASOF_LABEL, True), (ZEP_RAW_LABEL, False)),
            True,
        ),
        (lambda: LightRAGMemory(cfg, inf_engine), ((LIGHTRAG_LABEL, False),), False),
    )
    judge = AnswerJudge(inf_engine)

    detailed_rows: List[Dict[str, Any]] = cfg.load_checkpoint_rows(
        cfg.exp2_csv_path, required_column="fuente_dataset"
    )
    # Los casos con error se reintentan una vez más al reanudar
    detailed_rows = [r for r in detailed_rows if r.get("estado") == "ok"]
    done = {(r["metodo"], r["case_id"]) for r in detailed_rows}
    csv_out_path = cfg.base_dir / cfg.exp2_csv_path

    consecutive_failures = [0 for _ in systems]
    group_list = list(groups.values())
    batch_size = max(1, cfg.exp2_parallel_groups)
    progress = tqdm(total=len(group_list), desc="Experimento 2 (grupos de casos)")
    for start in range(0, len(group_list), batch_size):
        batch = group_list[start : start + batch_size]
        for system_idx, (make_system, arms, parallel) in enumerate(systems):
            pendings = [
                [
                    (metodo, as_of, case)
                    for case in group_cases
                    for metodo, as_of in arms
                    if (metodo, case.case_id) not in done
                ]
                for group_cases in batch
            ]
            calls = [
                _evaluate_group(make_system(), arms, judge, pending)
                for pending in pendings
                if pending
            ]
            if parallel:
                results = await asyncio.gather(*calls)
            else:
                results = [await call for call in calls]
            for group_rows in results:
                detailed_rows.extend(group_rows)
                failed = all(row["estado"] == "error" for row in group_rows)
                consecutive_failures[system_idx] = (
                    consecutive_failures[system_idx] + 1 if failed else 0
                )

        # Guardado incremental por lote para tolerancia a fallos en Colab
        pd.DataFrame(detailed_rows).to_csv(csv_out_path, index=False, encoding="utf-8")
        cfg.sync_to_google_drive(csv_out_path)
        progress.update(len(batch))

        if max(consecutive_failures) >= MAX_CONSECUTIVE_FAILED_GROUPS:
            raise RuntimeError(
                f"Un sistema falló en {max(consecutive_failures)} grupos seguidos. "
                f"Último error: {detailed_rows[-1]['juez_rationale']}"
            )
    progress.close()

    df_detailed = pd.DataFrame(detailed_rows)
    df_valid = df_detailed[df_detailed["juez_fuente"] == "llm"].copy()
    df_valid["correcto"] = df_valid["correcto"].astype(float)
    df_summary = df_valid.groupby(["fuente_dataset", "metodo"], as_index=False).agg(
        filas_evaluadas=("case_id", "count"),
        accuracy=("correcto", "mean"),
    )

    print("\n=== RESUMEN EXPERIMENTO 2 (CONSISTENCIA BI-TEMPORAL) ===")
    print(df_summary.to_string(index=False))
    unevaluated = len(df_detailed) - len(df_valid)
    if unevaluated:
        print(
            f"[AVISO] {unevaluated}/{len(df_detailed)} filas sin evaluar por errores "
            "o por falta de veredicto del juez (ver resumen_estadistico.csv)."
        )
    print(f"\n[OK] Resultados detallados exportados a: {csv_out_path.resolve()}")

    return df_detailed, df_summary


if __name__ == "__main__":
    asyncio.run(run_experiment_2())
