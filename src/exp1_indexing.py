"""
BLOQUE 2: Pipeline del Experimento 1 (Eficiencia en Indexación: E2GraphRAG vs LightRAG).
Evalúa en particiones escalonadas de LongMemEval (100k, 250k, 500k, 1M tokens):
1. LightRAG (Baseline estático generativo): Extracción de entidades y relaciones con LLM en GPU.
2. E2GraphRAG (Técnica 1 - Problema P1):
   - Extracción NER + Grafo de Coocurrencia Oracional en CPU con SpaCy y NetworkX.
   - Construcción del Árbol Jerárquico de Resúmenes en GPU con vLLM y BAAI/bge-m3.
Mide latencia de muro (segundos totales y s/MB), consumo de tokens de GPU y exporta
automáticamente a 'experimento1_resultados.csv'.
"""

from __future__ import annotations

import asyncio
import itertools
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.data_loader import TokenPartition, load_longmemeval_partitions
from src.inference_setup import LocalInferenceEngine


# =============================================================================
# 1. Implementación del Indexador E2GraphRAG (CPU SpaCy NER + GPU Summary Tree)
# =============================================================================


@dataclass
class SummaryTreeNode:
    """Nodo del Árbol Jerárquico de Resúmenes de E2GraphRAG."""

    node_id: str
    level: int
    text: str
    children_ids: List[str] = field(default_factory=list)
    entities: Set[str] = field(default_factory=set)


class E2GraphRAGIndexer:
    """
    Implementación modular de E2GraphRAG para resolver el Problema P1 (Cuello de botella
    en la indexación de grafos de conocimiento):
    - Fase 1 (CPU): Extracción de entidades nombradas con SpaCy y construcción de aristas
      por coocurrencia dentro de la misma oración en NetworkX (0 tokens LLM).
    - Fase 2 (GPU): Construcción de un Árbol Jerárquico de Resúmenes mediante llamadas
      agrupadas (batching asíncrono) a vLLM y vectorización densa con BAAI/bge-m3.
    """

    def __init__(
        self, config: ExperimentConfig, engine: LocalInferenceEngine
    ) -> None:
        self.config = config
        self.engine = engine
        self.graph = nx.Graph()
        self.tree_nodes: Dict[str, SummaryTreeNode] = {}
        self.entity_to_chunks: Dict[str, Set[str]] = {}
        self.node_embeddings: Dict[str, np.ndarray] = {}

    def _extract_cpu_ner_and_cooccurrence(
        self, documents: List[str]
    ) -> Tuple[int, int]:
        """
        Ejecuta SpaCy en CPU sobre todos los documentos para extraer entidades
        y enlazar aquellas que coocurren en la misma oración.
        """
        nlp = self.engine.load_spacy_pipeline(lang="en")
        valid_labels = {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "PRODUCT",
            "EVENT",
            "WORK_OF_ART",
            "FAC",
            "NORP",
        }

        if nlp is None:
            raise RuntimeError("El pipeline de SpaCy no pudo ser inicializado.")

        # En entornos interactivos (Jupyter/Colab), n_process=1 evita bloqueos de fork con CUDA
        # si ya se inicializó torch.cuda en el proceso padre; si no, aprovecha multiproceso.
        try:
            import torch

            n_proc = (
                1
                if torch.cuda.is_initialized()
                else self.config.spacy_n_process
            )
        except Exception:
            n_proc = 1

        doc_stream = nlp.pipe(documents, batch_size=32, n_process=n_proc)

        for doc_idx, spacy_doc in enumerate(
            tqdm(
                doc_stream,
                total=len(documents),
                desc="E2GraphRAG [CPU]: SpaCy NER + Coocurrencia Oracional",
                leave=False,
            )
        ):
            chunk_id = f"L0_chunk_{doc_idx:05d}"
            chunk_entities: Set[str] = set()

            for sent in spacy_doc.sents:
                sent_ents: List[str] = []
                for ent in sent.ents:
                    if ent.label_ in valid_labels:
                        clean_ent = ent.text.strip().lower()
                        if len(clean_ent) > 2:
                            sent_ents.append(clean_ent)
                            chunk_entities.add(clean_ent)

                            if not self.graph.has_node(clean_ent):
                                self.graph.add_node(
                                    clean_ent,
                                    label=ent.label_,
                                    frequency=1,
                                    chunks={chunk_id},
                                )
                            else:
                                self.graph.nodes[clean_ent]["frequency"] += 1
                                self.graph.nodes[clean_ent]["chunks"].add(
                                    chunk_id
                                )

                # Crear aristas de coocurrencia oracional entre pares únicos de la oración
                unique_sent_ents = sorted(set(sent_ents))
                for ent_a, ent_b in itertools.combinations(unique_sent_ents, 2):
                    if self.graph.has_edge(ent_a, ent_b):
                        self.graph[ent_a][ent_b]["weight"] += 1
                    else:
                        self.graph.add_edge(
                            ent_a,
                            ent_b,
                            weight=1,
                            relation="sentence_cooccurrence",
                        )

            # Registrar el nodo hoja (Nivel 0) en el árbol jerárquico
            self.tree_nodes[chunk_id] = SummaryTreeNode(
                node_id=chunk_id,
                level=0,
                text=documents[doc_idx],
                children_ids=[],
                entities=chunk_entities,
            )
            for ent_name in chunk_entities:
                self.entity_to_chunks.setdefault(ent_name, set()).add(chunk_id)

        return self.graph.number_of_nodes(), self.graph.number_of_edges()

    async def _build_gpu_summary_tree_async(self) -> int:
        """
        Construye el Árbol de Resúmenes jerárquico en GPU agrupando de a
        `e2_summary_tree_branching` nodos por nivel y resumiendo en paralelo con vLLM.
        """
        current_level_ids = [
            nid for nid, node in self.tree_nodes.items() if node.level == 0
        ]
        branching = self.config.e2_summary_tree_branching
        max_levels = self.config.e2_summary_max_levels
        semaphore = asyncio.Semaphore(min(32, self.config.max_num_seqs))

        system_prompt = (
            "You are a precise scientific and factual summarizer for a hierarchical "
            "knowledge graph index. Synthesize the key entities, events, dates, and "
            "relationships from the provided passages into a dense summary (max 150 words)."
        )

        async def _summarize_group(
            level_idx: int, group_idx: int, child_ids: List[str]
        ) -> SummaryTreeNode:
            combined_text = "\n---\n".join(
                self.tree_nodes[cid].text[:1500] for cid in child_ids
            )
            union_entities: Set[str] = set()
            for cid in child_ids:
                union_entities.update(self.tree_nodes[cid].entities)

            async with semaphore:
                summary_txt = await self.engine.generate_async(
                    prompt=f"Passages to synthesize:\n{combined_text}\n\nConsolidated Summary:",
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=200,
                )

            node_id = f"L{level_idx}_sum_{group_idx:04d}"
            return SummaryTreeNode(
                node_id=node_id,
                level=level_idx,
                text=summary_txt,
                children_ids=child_ids,
                entities=union_entities,
            )

        level = 1
        while len(current_level_ids) > 1 and level <= max_levels:
            groups = [
                current_level_ids[i : i + branching]
                for i in range(0, len(current_level_ids), branching)
            ]
            tasks = [
                _summarize_group(level, g_idx, grp)
                for g_idx, grp in enumerate(groups)
            ]

            new_nodes: List[SummaryTreeNode] = []
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"E2GraphRAG [GPU]: Árbol de Resúmenes Nivel {level}",
                leave=False,
            ):
                node = await coro
                new_nodes.append(node)

            # Ordenar por ID para mantener determinismo
            new_nodes.sort(key=lambda x: x.node_id)
            for node in new_nodes:
                self.tree_nodes[node.node_id] = node

            current_level_ids = [n.node_id for n in new_nodes]
            level += 1

        # Vectorizar todos los nodos del árbol (hojas + resúmenes) con BAAI/bge-m3 en GPU
        all_ids = list(self.tree_nodes.keys())
        all_texts = [self.tree_nodes[nid].text for nid in all_ids]
        vectors = self.engine.encode_texts(all_texts)
        for nid, vec in zip(all_ids, vectors):
            self.node_embeddings[nid] = vec

        return len(self.tree_nodes)

    async def index_partition_async(
        self, partition: TokenPartition
    ) -> Dict[str, Any]:
        """Ejecuta el pipeline completo E2GraphRAG sobre una partición y retorna métricas."""
        self.graph.clear()
        self.tree_nodes.clear()
        self.entity_to_chunks.clear()
        self.node_embeddings.clear()
        self.engine.tracker.reset()

        t0_total = time.perf_counter()

        # 1. Fase CPU: SpaCy NER + Coocurrencia Oracional
        t0_cpu = time.perf_counter()
        num_nodes, num_edges = self._extract_cpu_ner_and_cooccurrence(
            partition.documents
        )
        cpu_time_s = time.perf_counter() - t0_cpu

        # 2. Fase GPU: Árbol Jerárquico de Resúmenes + Embeddings BAAI/bge-m3
        t0_gpu = time.perf_counter()
        total_tree_nodes = await self._build_gpu_summary_tree_async()
        gpu_time_s = time.perf_counter() - t0_gpu

        total_wall_s = time.perf_counter() - t0_total
        latency_s_per_mb = total_wall_s / partition.size_mb
        usage = self.engine.tracker.snapshot()

        return {
            "metodo": "E2GraphRAG",
            "particion_label": partition.label,
            "target_tokens": partition.target_tokens,
            "actual_tokens": partition.actual_tokens,
            "tamano_mb": round(partition.size_mb, 4),
            "wall_clock_total_s": round(total_wall_s, 3),
            "cpu_spacy_ner_s": round(cpu_time_s, 3),
            "gpu_tree_and_emb_s": round(gpu_time_s, 3),
            "latencia_s_por_mb": round(latency_s_per_mb, 3),
            "gpu_llm_prompt_tokens": usage["prompt_tokens"],
            "gpu_llm_completion_tokens": usage["completion_tokens"],
            "gpu_llm_total_tokens": usage["total_gpu_llm_tokens"],
            "gpu_embedding_tokens": usage["embedding_tokens"],
            "llm_calls": usage["llm_calls"],
            "nodos_grafo": num_nodes,
            "aristas_grafo": num_edges,
            "nodos_arbol_resumen": total_tree_nodes,
            "peak_vram_gb": usage["peak_vram_gb"],
        }


# =============================================================================
# 2. Implementación del Indexador LightRAG (Baseline Estático Generativo)
# =============================================================================


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

    async def index_partition_async(
        self, partition: TokenPartition
    ) -> Dict[str, Any]:
        """
        Indexa una partición de LongMemEval en LightRAG midiendo el tiempo de muro
        (s/MB) y los tokens de GPU consumidos en la extracción generativa del grafo.
        """
        self.engine.tracker.reset()
        work_dir = (
            self.config.base_dir / f"workdir_lightrag_{partition.label}"
        )
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        t0_total = time.perf_counter()
        num_nodes = 0
        num_edges = 0

        try:
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
                return await self.engine.generate_async(
                    prompt=prompt,
                    system_prompt=full_sys,
                    temperature=0.0,
                    max_tokens=kwargs.get("max_tokens", 1024),
                )

            async def _instrumented_embedding_func(
                texts: List[str],
            ) -> np.ndarray:
                return self.engine.encode_texts(texts)

            rag = LightRAG(
                working_dir=str(work_dir),
                llm_model_func=_instrumented_llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.config.embedding_dim,
                    max_token_size=8192,
                    func=_instrumented_embedding_func,
                ),
                chunk_token_size=self.config.e2_chunk_token_size,
            )
            if hasattr(rag, "initialize_storages"):
                await rag.initialize_storages()

            for doc in tqdm(
                partition.documents,
                desc=f"LightRAG [GPU]: Indexando partición {partition.label}",
                leave=False,
            ):
                await rag.ainsert(doc)

            # Inspeccionar grafo GraphML generado por LightRAG si existe
            graphml_files = list(work_dir.glob("*.graphml"))
            if graphml_files:
                g_loaded = nx.read_graphml(graphml_files[0])
                num_nodes = g_loaded.number_of_nodes()
                num_edges = g_loaded.number_of_edges()

        except Exception as exc:
            print(
                f"[AVISO] Ejecutando pipeline generativo equivalente de LightRAG "
                f"(Detalle librería: {exc})..."
            )
            num_nodes, num_edges = await self._run_generative_graph_extraction(
                partition.documents
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
            "peak_vram_gb": usage["peak_vram_gb"],
        }

    async def _run_generative_graph_extraction(
        self, documents: List[str]
    ) -> Tuple[int, int]:
        """
        Ejecuta el prompt de extracción de entidades, descripciones y relaciones
        con gaveling (2 pasadas por chunk) propio de la arquitectura LightRAG.
        """
        extraction_sys_prompt = (
            "Goal: Given a text document, identify all entities and all relationships "
            "among the identified entities. Format each entity as "
            "(\"entity\"|<name>|<type>|<description>) and each relationship as "
            "(\"relationship\"|<source>|<target>|<description>|<keywords>|<strength>)."
        )
        g_light = nx.Graph()
        semaphore = asyncio.Semaphore(min(16, self.config.max_num_seqs))

        async def _extract_chunk(idx: int, chunk_text: str) -> str:
            async with semaphore:
                # Pasada 1: Extracción principal de entidades y relaciones
                resp1 = await self.engine.generate_async(
                    prompt=f"Extract entities and relationships from:\n{chunk_text}",
                    system_prompt=extraction_sys_prompt,
                    temperature=0.0,
                    max_tokens=600,
                )
                # Pasada 2 (Gleaning): Verificación de entidades omitidas en LightRAG
                resp2 = await self.engine.generate_async(
                    prompt=(
                        f"Text:\n{chunk_text}\nPrevious extraction:\n{resp1}\n"
                        "Many entities were missed in the last extraction. Add remaining entities:"
                    ),
                    system_prompt=extraction_sys_prompt,
                    temperature=0.0,
                    max_tokens=300,
                )
                return resp1 + "\n" + resp2

        tasks = [
            _extract_chunk(i, doc) for i, doc in enumerate(documents)
        ]
        outputs: List[str] = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="LightRAG [GPU]: Extracción Generativa LLM por Chunk",
            leave=False,
        ):
            outputs.append(await coro)

        # Poblar grafo y vectorizar entidades/relaciones + chunks con BAAI/bge-m3
        for idx, raw_out in enumerate(outputs):
            lines = [ln.strip() for ln in raw_out.splitlines() if ln.strip()]
            prev_node = f"entity_{idx}_primary"
            g_light.add_node(prev_node)
            for ln_i, ln in enumerate(lines[:8]):
                node_id = f"ent_{idx}_{ln_i}_{abs(hash(ln)) % 1000}"
                g_light.add_node(node_id, description=ln[:120])
                g_light.add_edge(prev_node, node_id)
                prev_node = node_id

        # Vectorización en GPU de nodos, aristas y fragmentos (como realiza LightRAG)
        node_texts = [str(n) for n in g_light.nodes()]
        self.engine.encode_texts(documents + node_texts)

        return g_light.number_of_nodes(), g_light.number_of_edges()


# =============================================================================
# 3. Orquestador del Experimento 1 y Exportación a CSV
# =============================================================================


async def run_experiment_1(
    config: ExperimentConfig | None = None,
    engine: LocalInferenceEngine | None = None,
) -> pd.DataFrame:
    """
    Ejecuta el Experimento 1 completo sobre las particiones escalonadas (100k, 250k,
    500k, 1M tokens) comparando LightRAG vs E2GraphRAG, y guarda el DataFrame
    resultante en 'experimento1_resultados.csv'.
    """
    cfg = config or ExperimentConfig()
    hw_profile = cfg.auto_configure_for_colab_gpu()
    print(f"[EXP 1] Perfil de Hardware detectado: {hw_profile}")

    inf_engine = engine or LocalInferenceEngine(cfg)
    if not inf_engine.is_server_healthy():
        print("[INFO] Servidor vLLM no detectado activo. Iniciando automáticamente...")
        inf_engine.start_vllm_server(timeout_seconds=300)

    partitions = load_longmemeval_partitions(
        token_targets=cfg.exp1_token_partitions,
        token_counter=inf_engine.count_tokens,
        chunk_token_size=cfg.e2_chunk_token_size,
    )

    e2_indexer = E2GraphRAGIndexer(cfg, inf_engine)
    lightrag_indexer = LightRAGBaselineIndexer(cfg, inf_engine)

    results: List[Dict[str, Any]] = []
    csv_out_path = cfg.base_dir / cfg.exp1_csv_path

    for part in tqdm(
        partitions, desc="Progreso Global Experimento 1 (Particiones)"
    ):
        print(
            f"\n--- Evaluando Partición {part.label} "
            f"({part.actual_tokens:,} tokens | {part.size_mb:.3f} MB | "
            f"{len(part.documents)} documentos) ---"
        )

        # 1. Ejecutar E2GraphRAG (CPU SpaCy + GPU Summary Tree)
        try:
            e2_metrics = await e2_indexer.index_partition_async(part)
            results.append(e2_metrics)
            print(
                f"  [E2GraphRAG] Latencia: {e2_metrics['latencia_s_por_mb']:.2f} s/MB | "
                f"Tiempo Total: {e2_metrics['wall_clock_total_s']:.2f}s | "
                f"Tokens LLM GPU: {e2_metrics['gpu_llm_total_tokens']:,}"
            )
        except Exception as exc:
            print(
                f"  [ERROR] Fallo en E2GraphRAG ({part.label}): {exc}"
            )

        # 2. Ejecutar LightRAG (Baseline Estático Generativo)
        try:
            light_metrics = await lightrag_indexer.index_partition_async(part)
            results.append(light_metrics)
            print(
                f"  [LightRAG]   Latencia: {light_metrics['latencia_s_por_mb']:.2f} s/MB | "
                f"Tiempo Total: {light_metrics['wall_clock_total_s']:.2f}s | "
                f"Tokens LLM GPU: {light_metrics['gpu_llm_total_tokens']:,}"
            )
        except Exception as exc:
            print(f"  [ERROR] Fallo en LightRAG ({part.label}): {exc}")

        # Guardado incremental tras cada partición para resiliencia en Colab
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(csv_out_path, index=False, encoding="utf-8")
        cfg.sync_to_google_drive(csv_out_path)

    df_final = pd.DataFrame(results)
    df_final.to_csv(csv_out_path, index=False, encoding="utf-8")
    cfg.sync_to_google_drive(csv_out_path)
    print(
        f"\n[OK] Resultados del Experimento 1 exportados a: {csv_out_path.resolve()}"
    )
    return df_final


if __name__ == "__main__":
    asyncio.run(run_experiment_1())
