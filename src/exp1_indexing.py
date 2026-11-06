"""
BLOQUE 2: Pipeline del Experimento 1 (Eficiencia en Indexación: E2GraphRAG vs LightRAG).
Evalúa en particiones escalonadas de LongMemEval (100k a 1M tokens), con varias corridas:
1. LightRAG (Baseline estático generativo): Extracción de entidades y relaciones con LLM en GPU.
2. E2GraphRAG (Técnica 1 - Problema P1):
   - Portado del repositorio oficial en src/e2graphrag.py: grafo de coocurrencia en CPU con
     SpaCy y árbol de resúmenes en GPU con vLLM y BAAI/bge-m3.
Mide latencia de muro (segundos totales y s/MB) y tokens LLM, y exporta a
'experimento1_resultados.csv'. Además mide la calidad de respuesta sobre ambos índices en una
partición, porque un índice más barato solo es mejor si sigue siendo igual de útil
('experimento1_calidad.csv').
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.data_loader import (
    TokenPartition,
    load_longmemeval_partitions,
    load_quality_questions,
)
from src.e2graphrag import E2GraphRAGIndexer
from src.evaluation import AnswerJudge
from src.inference_setup import LocalInferenceEngine


MAX_INDEX_LLM_FAILURE_RATE = 0.05

ANSWER_SYSTEM_PROMPT = (
    "You answer questions about a user's past conversations using ONLY the given context. "
    "Be direct and precise. If the context does not contain the answer, say you do not know."
)


# =============================================================================
# 1. Implementación del Indexador LightRAG (Baseline Estático Generativo)
# =============================================================================


async def create_lightrag(
    config: ExperimentConfig,
    engine: LocalInferenceEngine,
    work_dir: Path,
    workspace: str,
) -> Any:
    """
    Crea una instancia de LightRAG aislada. LightRAG conserva en memoria el estado de los
    documentos por workspace aunque se borre el directorio: sin un workspace propio por
    índice, los documentos ya vistos se omiten como duplicados y su costo no se mide.
    La caché del LLM se desactiva para que cada corrida pague su costo real.
    """
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def _instrumented_llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list = [],
        **kwargs: Any,
    ) -> str:
        full_sys = system_prompt or ""
        if history_messages:
            hist_str = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in history_messages
            )
            full_sys = f"{full_sys}\n{hist_str}".strip()
        return await engine.generate_async(
            prompt=prompt,
            system_prompt=full_sys,
            temperature=0.0,
            max_tokens=kwargs.get("max_tokens", 4096),
        )

    async def _instrumented_embedding_func(texts: List[str]) -> np.ndarray:
        return engine.encode_texts(texts)

    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(work_dir),
        workspace=re.sub(r"[^a-zA-Z0-9_]", "_", workspace),
        llm_model_func=_instrumented_llm_func,
        # La misma concurrencia máxima de llamadas al LLM que E2GraphRAG
        llm_model_max_async=min(32, config.max_num_seqs),
        max_parallel_insert=8,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        embedding_func=EmbeddingFunc(
            embedding_dim=config.embedding_dim,
            max_token_size=8192,
            func=_instrumented_embedding_func,
        ),
        chunk_token_size=config.e2_chunk_token_size,
        chunk_overlap_token_size=config.chunk_overlap_token_size,
    )
    await rag.initialize_storages()
    return rag


class LightRAGBaselineIndexer:
    """
    Envoltorio instrumentado de LightRAG (lightrag-hku) conectado al motor local
    vLLM/Ollama y BAAI/bge-m3. Registra cada token de GPU consumido durante la
    extracción generativa de entidades y relaciones.
    """

    def __init__(
        self, config: ExperimentConfig, engine: LocalInferenceEngine
    ) -> None:
        self.config = config
        self.engine = engine
        self.rag: Any = None

    async def index_partition_async(
        self, partition: TokenPartition, run_id: int = 1
    ) -> Dict[str, Any]:
        """
        Indexa una partición de LongMemEval en LightRAG midiendo el tiempo de muro
        (s/MB) y los tokens de GPU consumidos en la extracción generativa del grafo.
        """
        await self.close()
        workspace = f"{partition.label}_run{run_id}"
        work_dir = self.config.base_dir / f"workdir_lightrag_{workspace}"
        self.engine.tracker.reset()

        failures_before = self.engine.llm_failure_count
        t0_total = time.perf_counter()

        self.rag = await create_lightrag(
            self.config, self.engine, work_dir, workspace
        )
        await self.rag.ainsert(partition.documents)

        # El grafo se escribe en <working_dir>/<workspace>/
        graphml_files = list(work_dir.rglob("*.graphml"))
        num_nodes = 0
        num_edges = 0
        if graphml_files:
            g_loaded = nx.read_graphml(graphml_files[0])
            num_nodes = g_loaded.number_of_nodes()
            num_edges = g_loaded.number_of_edges()

        # LightRAG captura los errores del LLM y marca el documento como fallido sin
        # propagar la excepción: unos pocos se reportan, pero muchos harían incomparable
        # el costo medido.
        new_failures = self.engine.llm_failure_count - failures_before
        attempts = new_failures + self.engine.tracker.snapshot()["llm_calls"]
        if new_failures > MAX_INDEX_LLM_FAILURE_RATE * attempts:
            raise RuntimeError(
                f"LightRAG ({partition.label}): {new_failures}/{attempts} llamadas al LLM "
                f"fallaron. Último error: {self.engine.last_llm_error!r}"
            )
        if num_nodes == 0:
            raise RuntimeError(
                f"LightRAG ({partition.label}) terminó con un grafo vacío: "
                "la extracción de entidades no produjo resultados."
            )

        total_wall_s = time.perf_counter() - t0_total
        latency_s_per_mb = total_wall_s / partition.size_mb
        usage = self.engine.tracker.snapshot()

        return {
            "metodo": "LightRAG",
            "particion_label": partition.label,
            "target_tokens": partition.target_tokens,
            "actual_tokens": partition.actual_tokens,
            "tamano_mb": round(partition.size_mb, 4),
            "wall_clock_total_s": round(total_wall_s, 3),
            "cpu_spacy_ner_s": 0.0,
            "gpu_tree_and_emb_s": round(total_wall_s, 3),
            "latencia_s_por_mb": round(latency_s_per_mb, 3),
            "gpu_llm_prompt_tokens": usage["prompt_tokens"],
            "gpu_llm_completion_tokens": usage["completion_tokens"],
            "gpu_llm_total_tokens": usage["total_gpu_llm_tokens"],
            "gpu_embedding_tokens": usage["embedding_tokens"],
            "llm_calls": usage["llm_calls"],
            "nodos_grafo": num_nodes,
            "aristas_grafo": num_edges,
            "nodos_arbol_resumen": 0,
            "llm_llamadas_fallidas": new_failures,
            "peak_vram_embeddings_gb": usage["peak_vram_gb"],
        }

    async def answer(self, question: str) -> str:
        """Responde con la consulta híbrida propia de LightRAG sobre el último índice."""
        from lightrag import QueryParam

        failures_before = self.engine.llm_failure_count
        answer = await self.rag.aquery(
            question,
            param=QueryParam(mode="mix", enable_rerank=False, response_type="Single Paragraph"),
        )
        if self.engine.llm_failure_count > failures_before:
            raise RuntimeError(
                f"LightRAG no pudo responder: {self.engine.last_llm_error!r}"
            )
        return str(answer)

    async def close(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()
            self.rag = None


# =============================================================================
# 2. Orquestador del Experimento 1 y Exportación a CSV
# =============================================================================


async def run_quality_evaluation(
    cfg: ExperimentConfig,
    engine: LocalInferenceEngine,
    partitions: List[TokenPartition],
    e2_indexer: E2GraphRAGIndexer,
    lightrag_indexer: LightRAGBaselineIndexer,
) -> None:
    """
    Reindexa una partición (sin medir tiempos) y responde preguntas de LongMemEval
    cuyas sesiones de evidencia están en ella.
    """
    partition = next(
        p for p in partitions if p.target_tokens == cfg.exp1_quality_partition
    )
    questions = load_quality_questions(
        partition.session_ids, cfg.exp1_quality_questions, pilot=cfg.pilot
    )
    rows: List[Dict[str, Any]] = cfg.load_checkpoint_rows(
        cfg.exp1_quality_csv_path, required_column="juez_fuente"
    )
    # Las preguntas con error se reintentan al reanudar
    rows = [r for r in rows if r["juez_fuente"] != "error"]
    done = {(r["metodo"], r["question_id"]) for r in rows}
    csv_out_path = cfg.base_dir / cfg.exp1_quality_csv_path
    judge = AnswerJudge(engine)
    print(
        f"\n--- Calidad de respuesta: {len(questions)} preguntas sobre {partition.label} ---"
    )

    for metodo, indexer in (("E2GraphRAG", e2_indexer), ("LightRAG", lightrag_indexer)):
        pending = [q for q in questions if (metodo, q.question_id) not in done]
        if not pending:
            continue
        try:
            if metodo == "LightRAG":
                await indexer.index_partition_async(partition, run_id=0)
            else:
                await indexer.index_partition_async(partition)
        except Exception as exc:
            if not engine.is_server_healthy():
                raise
            print(f"[ERROR] Calidad {metodo}: no se pudo indexar: {exc}")
            continue
        for question in tqdm(pending, desc=f"Calidad {metodo}", leave=False):
            base_row = {
                "metodo": metodo,
                "particion_label": partition.label,
                "question_id": question.question_id,
                "question_type": question.question_type,
                "pregunta": question.question,
                "ground_truth": question.answer,
            }
            try:
                answer = await indexer.answer(question.question)
                verdict = await judge.evaluate(question.question, question.answer, answer)
                rows.append(
                    {
                        **base_row,
                        "respuesta_generada": answer,
                        "correcto": verdict["correct"],
                        "juez_fuente": verdict["source"],
                        "juez_rationale": verdict["rationale"],
                    }
                )
            except Exception as exc:
                if not engine.is_server_healthy():
                    raise
                rows.append(
                    {
                        **base_row,
                        "juez_fuente": "error",
                        "juez_rationale": f"{exc.__class__.__name__}: {exc}"[:500],
                    }
                )
            pd.DataFrame(rows).to_csv(csv_out_path, index=False, encoding="utf-8")
        cfg.sync_to_google_drive(csv_out_path)
    await lightrag_indexer.close()


async def run_experiment_1(
    config: ExperimentConfig | None = None,
    engine: LocalInferenceEngine | None = None,
) -> pd.DataFrame:
    """
    Ejecuta el Experimento 1 sobre las particiones escalonadas, con varias corridas
    independientes, y guarda el DataFrame resultante en 'experimento1_resultados.csv'.
    """
    cfg = config or ExperimentConfig()
    hw_profile = cfg.auto_configure_for_colab_gpu()
    print(f"[EXP 1] Perfil de Hardware detectado: {hw_profile}")

    inf_engine = engine or LocalInferenceEngine(cfg)
    if not inf_engine.is_server_healthy():
        print("[INFO] Servidor vLLM no detectado activo. Iniciando automáticamente...")
        inf_engine.start_vllm_server(timeout_seconds=900)

    partitions = load_longmemeval_partitions(
        token_targets=cfg.exp1_token_partitions,
        token_counter=inf_engine.count_tokens,
        chunk_token_size=cfg.e2_chunk_token_size,
    )

    # Cargar modelos y calentar el LLM fuera de lo medido: de otro modo ese costo
    # recae solo en el primer sistema de la primera partición.
    inf_engine.load_embedding_model()
    inf_engine.load_spacy_pipeline(lang="en")
    await inf_engine.generate_async("Reply with OK.", max_tokens=4)

    e2_indexer = E2GraphRAGIndexer(cfg, inf_engine)
    lightrag_indexer = LightRAGBaselineIndexer(cfg, inf_engine)

    results: List[Dict[str, Any]] = cfg.load_checkpoint_rows(
        cfg.exp1_csv_path, required_column="corrida"
    )
    done = {
        (r.get("metodo"), str(r.get("particion_label")), int(r.get("corrida")))
        for r in results
    }
    csv_out_path = cfg.base_dir / cfg.exp1_csv_path

    failed_measurements: List[str] = []

    async def _measure(metodo: str, part: TokenPartition, run_id: int) -> None:
        if (metodo, str(part.label), run_id) in done:
            print(f"  [RESUME] {metodo} ({part.label}, corrida {run_id}) ya calculado.")
            return
        try:
            if metodo == "E2GraphRAG":
                metrics = await e2_indexer.index_partition_async(part)
            else:
                metrics = await lightrag_indexer.index_partition_async(part, run_id=run_id)
                await lightrag_indexer.close()
        except Exception as exc:
            # Sin servidor no hay nada que medir. Si no, se salta esta medición: el análisis
            # reporta n, y al reanudar se reintenta.
            if not inf_engine.is_server_healthy():
                raise
            failed_measurements.append(f"{metodo} {part.label} corrida {run_id}: {exc}")
            print(f"  [ERROR] {failed_measurements[-1]}")
            return
        metrics["corrida"] = run_id
        results.append(metrics)
        # Guardado incremental (local + Drive) para no perder progreso si la VM se cae
        pd.DataFrame(results).to_csv(csv_out_path, index=False, encoding="utf-8")
        cfg.sync_to_google_drive(csv_out_path)
        print(
            f"  [{metodo}] Corrida {run_id} | Latencia: {metrics['latencia_s_por_mb']:.2f} s/MB | "
            f"Tiempo Total: {metrics['wall_clock_total_s']:.2f}s | "
            f"Tokens LLM GPU: {metrics['gpu_llm_total_tokens']:,}"
        )

    for run_id in range(1, cfg.exp1_num_runs + 1):
        # Se alterna quién corre primero para no favorecer siempre al mismo sistema
        # con el estado de la GPU y la caché de prefijos de vLLM.
        methods = ["E2GraphRAG", "LightRAG"]
        if run_id % 2 == 0:
            methods.reverse()
        for part in tqdm(partitions, desc=f"Experimento 1: corrida {run_id}"):
            print(
                f"\n--- Corrida {run_id} | Partición {part.label} "
                f"({part.actual_tokens:,} tokens | {part.size_mb:.3f} MB | "
                f"{len(part.documents)} documentos) ---"
            )
            for metodo in methods:
                await _measure(metodo, part, run_id)

    await run_quality_evaluation(cfg, inf_engine, partitions, e2_indexer, lightrag_indexer)

    if not results:
        raise RuntimeError(f"Ninguna medición terminó: {failed_measurements}")
    if failed_measurements:
        print(f"[AVISO] Mediciones omitidas: {failed_measurements}")

    df_final = pd.DataFrame(results)
    print(
        f"\n[OK] Resultados del Experimento 1 exportados a: {csv_out_path.resolve()}"
    )
    return df_final


if __name__ == "__main__":
    asyncio.run(run_experiment_1())
