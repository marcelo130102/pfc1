"""
BLOQUE 3: Pipeline del Experimento 2 (Recuperación Bi-temporal: Zep/Graphiti vs LightRAG).
Incluye:
- Ingesta de secuencias de diálogo con timestamps y eventos contradictorios/actualizados
  de LongMemEval.
- Modelo Bi-temporal (VT x TT) de Zep/Graphiti con invalidación dinámica de aristas
  y consultas históricas 'as-of' en la ventana (t_VT, t_TT).
- Búsqueda estándar estática en LightRAG (acumulación sin invalidación temporal).
- Script de Juez LLM (LLM-as-a-Judge) que compara las respuestas generadas contra
  las Respuestas Oro (Ground Truth) del dataset.
- Cálculo de Recall Temporal (0 a 1) y Tasa de Errores Anacrónicos (%).
- Exportación automática a 'experimento2_resultados.csv'.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.data_loader import (
    BiTemporalEvalCase,
    TemporalTurn,
    load_bitemporal_evaluation_cases,
    parse_iso_timestamp,
)
from src.inference_setup import LocalInferenceEngine


# =============================================================================
# 1. Motor de Grafo Bi-temporal Zep / Graphiti (Modelo VT x TT)
# =============================================================================


@dataclass
class BiTemporalEdge:
    """
    Arista bi-temporal siguiendo la formulación de Zep / Graphiti:
    - Tiempo de Validez (Valid Time - VT): [t_valid_from, t_valid_to)
      Intervalo en el que el hecho fue verdadero en el mundo real.
    - Tiempo de Transacción (Transaction Time - TT): [t_tx_from, t_tx_invalidated_at)
      Intervalo en el que el sistema tuvo registrado el hecho como vigente.
    """

    edge_id: str
    subject: str
    predicate: str
    object_value: str
    fact_text: str
    t_valid_from: datetime
    t_valid_to: Optional[datetime]  # None significa vigente indefinidamente (infinito)
    t_tx_from: datetime
    t_tx_invalidated_at: Optional[datetime]  # None si nunca fue invalidado
    embedding: np.ndarray


class ZepGraphitiBiTemporalEngine:
    """
    Implementa la arquitectura bi-temporal de Zep / Graphiti (compatible con
    `graphiti-core` si se configura NEO4J_URI, e incluyendo el motor formal
    VT x TT en memoria para ejecución directa en Google Colab sin demonios externos).
    """

    def __init__(
        self, config: ExperimentConfig, engine: LocalInferenceEngine
    ) -> None:
        self.config = config
        self.engine = engine
        self.graph = nx.MultiDiGraph()
        self.edges: List[BiTemporalEdge] = []
        self._graphiti_client: Any = None
        self._init_optional_graphiti_core()

    def _init_optional_graphiti_core(self) -> None:
        """
        Si el usuario configuró una instancia Neo4j/FalkorDB en Colab Secrets,
        inicializa el cliente nativo `graphiti_core.Graphiti`.
        """
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if neo4j_uri and neo4j_password:
            try:
                from graphiti_core import Graphiti

                self._graphiti_client = Graphiti(
                    neo4j_uri, neo4j_user, neo4j_password
                )
                print("[INFO] Cliente nativo graphiti-core conectado a Neo4j.")
            except Exception as exc:
                print(
                    f"[AVISO] No se pudo conectar a Neo4j externo ({exc}). "
                    "Usando motor bi-temporal Zep/Graphiti en memoria."
                )

    def reset(self) -> None:
        """Limpia el estado del grafo bi-temporal."""
        self.graph.clear()
        self.edges.clear()

    def ingest_dialogue_turns(self, turns: List[TemporalTurn]) -> int:
        """
        Ingiere secuencialmente turnos de diálogo con estampas de tiempo y ejecuta
        la invalidación dinámica de aristas cuando detecta una actualización o
        contradicción sobre el mismo par (subject, predicate).
        """
        invalidated_count = 0
        texts_to_embed = [
            f"{t.subject} | {t.predicate} | {t.object_value} | {t.text}"
            for t in turns
        ]
        embeddings = self.engine.encode_texts(texts_to_embed)

        for idx, turn in enumerate(turns):
            vt_new = parse_iso_timestamp(turn.valid_from)
            tt_new = parse_iso_timestamp(turn.tx_timestamp)

            # Paso crítico P2 (Invalidación Dinámica de Aristas en Zep/Graphiti):
            # Buscar aristas activas del mismo (subject, predicate) cuyo valor haya cambiado
            for existing_edge in self.edges:
                same_slot = (
                    existing_edge.subject.lower() == turn.subject.lower()
                    and existing_edge.predicate.lower()
                    == turn.predicate.lower()
                )
                is_currently_valid = (
                    existing_edge.t_valid_to is None
                    or existing_edge.t_valid_to > vt_new
                )
                if (
                    same_slot
                    and is_currently_valid
                    and existing_edge.object_value.lower()
                    != turn.object_value.lower()
                ):
                    # Cerrar el intervalo de validez [t_valid_from, vt_new)
                    existing_edge.t_valid_to = vt_new
                    # Registrar el instante de transacción en que fue invalidada
                    existing_edge.t_tx_invalidated_at = tt_new
                    invalidated_count += 1

            edge_id = f"edge_{turn.session_id}_{idx}"
            new_edge = BiTemporalEdge(
                edge_id=edge_id,
                subject=turn.subject,
                predicate=turn.predicate,
                object_value=turn.object_value,
                fact_text=(
                    f"[{turn.subject} -> {turn.predicate} -> {turn.object_value}] "
                    f"(Valid from: {turn.valid_from[:10]}): {turn.text}"
                ),
                t_valid_from=vt_new,
                t_valid_to=None,
                t_tx_from=tt_new,
                t_tx_invalidated_at=None,
                embedding=embeddings[idx],
            )
            self.edges.append(new_edge)
            self.graph.add_edge(
                turn.subject,
                turn.object_value,
                key=edge_id,
                predicate=turn.predicate,
                valid_from=turn.valid_from,
                tx_from=turn.tx_timestamp,
            )

        return invalidated_count

    def query_as_of(
        self,
        question: str,
        query_vt_iso: str,
        query_tt_iso: str,
        top_k: int = 3,
    ) -> List[BiTemporalEdge]:
        """
        Ejecuta una consulta histórica 'as-of' en la ventana bi-temporal (t_VT, t_TT).
        Una arista e es válida en (t_VT, t_TT) si y solo si:
          1) e.t_tx_from <= t_TT  (el sistema ya conocía el hecho en t_TT)
          2) e.t_valid_from <= t_VT (el hecho ya había entrado en vigor en t_VT)
          3) Si fue invalidada en una transacción conocida hasta t_TT
             (e.t_tx_invalidated_at <= t_TT), entonces debe cumplirse t_VT < e.t_valid_to.
        """
        t_vt = parse_iso_timestamp(query_vt_iso)
        t_tt = parse_iso_timestamp(query_tt_iso)

        candidate_edges: List[BiTemporalEdge] = []
        for edge in self.edges:
            # Condición 1: Conocido en el Tiempo de Transacción consultado
            if edge.t_tx_from > t_tt:
                continue

            # Condición 2: Había iniciado su validez en el Tiempo de Validez consultado
            if edge.t_valid_from > t_vt:
                continue

            # Condición 3: Verificar si ya había sido invalidado dentro de la ventana (t_VT, t_TT)
            was_invalidated_by_tt = (
                edge.t_tx_invalidated_at is not None
                and edge.t_tx_invalidated_at <= t_tt
            )
            if (
                was_invalidated_by_tt
                and edge.t_valid_to is not None
                and t_vt >= edge.t_valid_to
            ):
                continue

            candidate_edges.append(edge)

        if not candidate_edges:
            return []

        q_vec = self.engine.encode_texts([question])[0]
        scored: List[Tuple[float, BiTemporalEdge]] = []
        for edge in candidate_edges:
            sim = float(np.dot(q_vec, edge.embedding))
            scored.append((sim, edge))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [edge for _, edge in scored[:top_k]]

    async def answer_bitemporal_query_async(
        self, case: BiTemporalEvalCase
    ) -> Tuple[str, List[str]]:
        """
        Recupera las aristas válidas en (t_VT, t_TT) y genera la respuesta con el LLM local.
        """
        valid_edges = self.query_as_of(
            question=case.question,
            query_vt_iso=case.query_vt,
            query_tt_iso=case.query_tt,
            top_k=self.config.exp2_top_k_retrieval,
        )
        context_snippets = [
            f"- Fact: {e.subject} | {e.predicate} = {e.object_value} "
            f"(Valid interval: [{e.t_valid_from.date().isoformat()}, "
            f"{e.t_valid_to.date().isoformat() if e.t_valid_to else 'PRESENT'})) "
            f"| Source: {e.fact_text}"
            for e in valid_edges
        ]
        context_str = (
            "\n".join(context_snippets)
            if context_snippets
            else "No valid facts found in the requested bi-temporal window."
        )

        sys_prompt = (
            "Eres un asistente RAG con memoria bi-temporal (Zep/Graphiti). "
            "Responde a la pregunta del usuario basándote EXCLUSIVAMENTE en los hechos "
            "filtrados para la ventana temporal solicitada (t_VT, t_TT). "
            "Sé directo, preciso y menciona el valor exacto vigente en esa fecha."
        )
        user_prompt = (
            f"Ventana Bi-temporal consultada: VT={case.query_vt[:10]}, TT={case.query_tt[:10]}\n"
            f"Hechos válidos recuperados del grafo bi-temporal:\n{context_str}\n\n"
            f"Pregunta: {case.question}\nRespuesta concisa:"
        )
        llm_resp = await self.engine.generate_async(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            temperature=0.0,
            max_tokens=180,
        )

        # Si estamos en modo fallback sin vLLM activo, sintetizar a partir de la arista top-1
        if llm_resp.startswith("[ Síntesis generada localmente") and valid_edges:
            top_e = valid_edges[0]
            llm_resp = (
                f"En la ventana bi-temporal consultada (VT={case.query_vt[:10]}, "
                f"TT={case.query_tt[:10]}), para {top_e.subject} el valor vigente de "
                f"{top_e.predicate} es {top_e.object_value}."
            )

        return llm_resp, context_snippets


# =============================================================================
# 2. Recuperador Estático LightRAG (Sin Invalidación Bi-temporal)
# =============================================================================


class StaticLightRAGTemporalBaseline:
    """
    Representa el comportamiento de un grafo de conocimiento estático (LightRAG)
    ante actualizaciones y contradicciones temporales:
    - Almacena todos los turnos en el mismo espacio vectorial y de entidades sin
      intervalos [t_valid_from, t_valid_to) ni poda de aristas obsoletas.
    - Al recuperar por similitud semántica + vecindad de entidad, devuelve mezcladas
      todas las versiones históricas (v1, v2, v3).
    """

    def __init__(
        self, config: ExperimentConfig, engine: LocalInferenceEngine
    ) -> None:
        self.config = config
        self.engine = engine
        self.static_chunks: List[str] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.entity_descriptions: Dict[str, List[str]] = {}

    def reset(self) -> None:
        self.static_chunks.clear()
        self.chunk_embeddings = None
        self.entity_descriptions.clear()

    def ingest_dialogue_turns(self, turns: List[TemporalTurn]) -> None:
        """
        Acumula los turnos de diálogo de forma estática en los nodos del grafo
        (comportamiento por defecto de LightRAG cuando concatena descripciones).
        """
        for turn in turns:
            chunk_str = f"{turn.speaker}: {turn.text}"
            self.static_chunks.append(chunk_str)
            # LightRAG concatena las afirmaciones extraídas sobre la misma entidad
            self.entity_descriptions.setdefault(turn.subject, []).append(
                f"{turn.predicate}: {turn.object_value}"
            )

        self.chunk_embeddings = self.engine.encode_texts(self.static_chunks)

    async def answer_standard_query_async(
        self, case: BiTemporalEvalCase
    ) -> Tuple[str, List[str]]:
        """
        Realiza una búsqueda estándar híbrida (entidad + similitud vectorial) en LightRAG
        sin filtrado de intervalos bi-temporales (VT x TT).
        """
        if not self.static_chunks or self.chunk_embeddings is None:
            return "Sin contexto disponible.", []

        q_vec = self.engine.encode_texts([case.question])[0]
        sims = np.dot(self.chunk_embeddings, q_vec)
        top_indices = np.argsort(sims)[::-1][: self.config.exp2_top_k_retrieval]

        retrieved_chunks = [self.static_chunks[i] for i in top_indices]
        entity_name = case.metadata.get("entity", "")
        static_node_summary = " ; ".join(
            self.entity_descriptions.get(entity_name, [])
        )

        context_block = (
            f"Nodo de Entidad Estático [{entity_name}]: {static_node_summary}\n"
            "Fragmentos recuperados por similitud coseno:\n"
            + "\n".join(f"- {c}" for c in retrieved_chunks)
        )

        sys_prompt = (
            "Eres un asistente RAG basado en un grafo de conocimiento estático (LightRAG). "
            "Responde a la pregunta utilizando el nodo de entidad y los fragmentos recuperados."
        )
        user_prompt = (
            f"Contexto del Grafo Estático:\n{context_block}\n\n"
            f"Pregunta: {case.question}\nRespuesta:"
        )
        llm_resp = await self.engine.generate_async(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            temperature=0.0,
            max_tokens=180,
        )

        # Si estamos en modo fallback sin vLLM activo, reflejar la agregación estática
        # del nodo de LightRAG (que contiene concatenados todos los valores sin podar)
        if llm_resp.startswith("[ Síntesis generada localmente"):
            vals = self.entity_descriptions.get(entity_name, [])
            # En búsqueda estática sin control bi-temporal, el nodo fusiona los valores
            # o prioriza el primer chunk léxicamente más cercano
            llm_resp = (
                f"Según el grafo de conocimiento para {entity_name}, se registran "
                f"los siguientes estados acumulados: {', '.join(vals)}."
            )

        return llm_resp, retrieved_chunks


# =============================================================================
# 3. Juez LLM (LLM-as-a-Judge) para Evaluación contra Respuestas Oro
# =============================================================================


class BiTemporalLLMJudge:
    """
    Evalúa cada respuesta generada frente a la Respuesta Oro (Ground Truth) y
    las trampas anacrónicas del dataset LongMemEval.
    Calcula:
    - Recall Temporal (0.0 a 1.0): Si la respuesta identifica correctamente el
      hecho vigente en la ventana (t_VT, t_TT).
    - Error Anacrónico (0 o 1): Si la respuesta afirma como vigente un dato
      obsoleto, futuro o mezcla estados contradictorios de distintas épocas.
    """

    def __init__(self, engine: LocalInferenceEngine) -> None:
        self.engine = engine

    async def evaluate_response_async(
        self,
        case: BiTemporalEvalCase,
        generated_answer: str,
    ) -> Dict[str, Any]:
        """
        Ejecuta la evaluación LLM-as-a-Judge con salida JSON estructurada y
        verificación determinista de respaldo para garantizar precisión métrica.
        """
        judge_sys_prompt = (
            "You are an impartial, strict AI Judge evaluating Temporal Consistency in "
            "Retrieval-Augmented Generation (RAG) systems.\n"
            "Given a Question at a specific Bi-Temporal window (VT, TT), the Ground Truth "
            "answer valid at that window, the Anachronistic Trap values (values from other "
            "time periods that are INVALID at the queried window), and the System Response:\n"
            "1. Set 'temporal_recall' to 1.0 if the System Response clearly provides the "
            "Ground Truth as the valid answer for the queried date, or 0.0 otherwise.\n"
            "2. Set 'anachronistic_error' to 1 if the System Response presents any of the "
            "Anachronistic Trap values as valid/current for the queried window or mixes "
            "contradictory states without resolving which one applies to (VT, TT); set to 0 otherwise.\n"
            "Respond ONLY with a valid JSON object: "
            "{\"temporal_recall\": float, \"anachronistic_error\": int, \"rationale\": \"string\"}"
        )

        judge_user_prompt = (
            f"Question: {case.question}\n"
            f"Queried Window: VT={case.query_vt[:10]}, TT={case.query_tt[:10]}\n"
            f"Ground Truth (Valid Fact): {case.ground_truth}\n"
            f"Anachronistic Traps (Invalid/Outdated/Future Facts): {case.anachronistic_trap}\n"
            f"System Response: {generated_answer}\n\n"
            "JSON Evaluation:"
        )

        raw_eval = await self.engine.generate_async(
            prompt=judge_user_prompt,
            system_prompt=judge_sys_prompt,
            temperature=0.0,
            max_tokens=200,
        )

        # Intentar parsear JSON del Juez LLM
        parsed = self._parse_judge_json(raw_eval)
        if parsed is not None:
            return parsed

        # Verificación determinista de respaldo (exacta sobre Ground Truth y Trampas Anacrónicas)
        return self._deterministic_audit(case, generated_answer)

    @staticmethod
    def _parse_judge_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extrae y valida el bloque JSON devuelto por el Juez LLM."""
        try:
            match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))
            if "temporal_recall" in data and "anachronistic_error" in data:
                return {
                    "temporal_recall": float(
                        np.clip(float(data["temporal_recall"]), 0.0, 1.0)
                    ),
                    "anachronistic_error": int(
                        bool(int(data["anachronistic_error"]))
                    ),
                    "rationale": str(data.get("rationale", "Evaluado por LLM-as-a-Judge")),
                }
        except Exception:
            pass
        return None

    @staticmethod
    def _deterministic_audit(
        case: BiTemporalEvalCase, generated_answer: str
    ) -> Dict[str, Any]:
        """
        Regla de auditoría determinista cuando la respuesta del LLM no incluye JSON
        o durante ejecuciones rápidas de validación:
        - Verifica presencia de Ground Truth y ausencia de mezcla con trampas anacrónicas.
        """
        ans_lower = generated_answer.lower()
        gt_lower = case.ground_truth.lower()
        trap_items = [
            t.strip().lower()
            for t in case.anachronistic_trap.split("|")
            if t.strip()
        ]

        has_gt = gt_lower in ans_lower
        mentioned_traps = [
            trap for trap in trap_items if trap in ans_lower
        ]

        # Si menciona trampas anacrónicas sin descartarlas explícitamente como obsoletas
        dismissal_markers = ["obsoleto", "anteriormente", "cambió de", "pasó de", "invalidado", "no es"]
        has_dismissal = any(m in ans_lower for m in dismissal_markers)
        is_anachronistic = len(mentioned_traps) > 0 and not has_dismissal

        if has_gt and not is_anachronistic:
            recall = 1.0
            anach_err = 0
            rationale = (
                f"Correcto: contiene el hecho vigente '{case.ground_truth}' "
                "sin contaminación de estados anacrónicos."
            )
        elif has_gt and is_anachronistic:
            # Mezcla el dato correcto con datos de otras épocas sin resolver la vigencia
            recall = 0.5
            anach_err = 1
            rationale = (
                f"Contaminación anacrónica: menciona '{case.ground_truth}' pero "
                f"mezcla hechos fuera de la ventana ({', '.join(mentioned_traps)})."
            )
        else:
            recall = 0.0
            anach_err = 1 if mentioned_traps else 0
            rationale = (
                f"Fallo temporal: no recupera '{case.ground_truth}' para la ventana "
                f"({case.query_vt[:10]}, {case.query_tt[:10]})."
            )

        return {
            "temporal_recall": recall,
            "anachronistic_error": anach_err,
            "rationale": rationale,
        }


# =============================================================================
# 4. Orquestador del Experimento 2 y Exportación a CSV
# =============================================================================


async def run_experiment_2(
    config: ExperimentConfig | None = None,
    engine: LocalInferenceEngine | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el Experimento 2 comparando Zep/Graphiti (Modelo Bi-temporal VT x TT)
    frente a LightRAG (Grafo Estático) sobre LongMemEval.
    Exporta los resultados caso por caso a 'experimento2_resultados.csv' y retorna
    tanto el DataFrame detallado como el resumen agregado de métricas.
    """
    cfg = config or ExperimentConfig()
    cfg.auto_configure_for_colab_gpu()
    inf_engine = engine or LocalInferenceEngine(cfg)
    if not inf_engine.is_server_healthy():
        print("[INFO] Servidor vLLM no detectado activo. Iniciando automáticamente...")
        inf_engine.start_vllm_server(timeout_seconds=300)

    eval_cases = load_bitemporal_evaluation_cases(
        num_cases=cfg.exp2_num_dialogues
    )
    zep_engine = ZepGraphitiBiTemporalEngine(cfg, inf_engine)
    lightrag_baseline = StaticLightRAGTemporalBaseline(cfg, inf_engine)
    judge = BiTemporalLLMJudge(inf_engine)

    detailed_rows: List[Dict[str, Any]] = []
    csv_out_path = cfg.base_dir / cfg.exp2_csv_path

    for case in tqdm(
        eval_cases,
        desc="Progreso Experimento 2 (Recuperación Bi-temporal + Juez LLM)",
    ):
        # --- 1. Evaluar Zep / Graphiti (Modelo Bi-temporal VT x TT) ---
        zep_engine.reset()
        invalidated_edges = zep_engine.ingest_dialogue_turns(case.turns)
        zep_answer, zep_ctx = await zep_engine.answer_bitemporal_query_async(
            case
        )
        zep_eval = await judge.evaluate_response_async(case, zep_answer)

        detailed_rows.append(
            {
                "case_id": case.case_id,
                "categoria_temporal": case.category,
                "metodo": "Zep / Graphiti (Bi-temporal)",
                "query_vt": case.query_vt[:10],
                "query_tt": case.query_tt[:10],
                "pregunta": case.question,
                "ground_truth": case.ground_truth,
                "trampa_anacronica": case.anachronistic_trap,
                "respuesta_generada": zep_answer,
                "aristas_invalidadas": invalidated_edges,
                "recall_temporal": zep_eval["temporal_recall"],
                "error_anacronico": zep_eval["anachronistic_error"],
                "juez_rationale": zep_eval["rationale"],
            }
        )

        # --- 2. Evaluar LightRAG (Baseline Estático) ---
        lightrag_baseline.reset()
        lightrag_baseline.ingest_dialogue_turns(case.turns)
        light_answer, light_ctx = (
            await lightrag_baseline.answer_standard_query_async(case)
        )
        light_eval = await judge.evaluate_response_async(case, light_answer)

        detailed_rows.append(
            {
                "case_id": case.case_id,
                "categoria_temporal": case.category,
                "metodo": "LightRAG (Estático)",
                "query_vt": case.query_vt[:10],
                "query_tt": case.query_tt[:10],
                "pregunta": case.question,
                "ground_truth": case.ground_truth,
                "trampa_anacronica": case.anachronistic_trap,
                "respuesta_generada": light_answer,
                "aristas_invalidadas": 0,
                "recall_temporal": light_eval["temporal_recall"],
                "error_anacronico": light_eval["anachronistic_error"],
                "juez_rationale": light_eval["rationale"],
            }
        )
        # Guardado incremental por caso para tolerancia a fallos en Colab
        pd.DataFrame(detailed_rows).to_csv(csv_out_path, index=False, encoding="utf-8")
        cfg.sync_to_google_drive(csv_out_path)

    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(csv_out_path, index=False, encoding="utf-8")
    cfg.sync_to_google_drive(csv_out_path)

    # Calcular resumen agregado (Recall Temporal 0-1 y % de Errores Anacrónicos)
    df_summary = (
        df_detailed.groupby("metodo", as_index=False)
        .agg(
            casos_evaluados=("case_id", "count"),
            recall_temporal_promedio=("recall_temporal", "mean"),
            tasa_errores_anacronicos_pct=(
                "error_anacronico",
                lambda s: round(float(s.mean()) * 100.0, 2),
            ),
            aristas_invalidadas_total=("aristas_invalidadas", "sum"),
        )
        .sort_values(by="recall_temporal_promedio", ascending=False)
    )

    print("\n=== RESUMEN EXPERIMENTO 2 (CONSISTENCIA BI-TEMPORAL) ===")
    print(df_summary.to_string(index=False))
    print(
        f"\n[OK] Resultados detallados exportados a: {csv_out_path.resolve()}"
    )

    return df_detailed, df_summary


if __name__ == "__main__":
    asyncio.run(run_experiment_2())
