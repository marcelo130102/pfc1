"""
E²GraphRAG: indexación y recuperación portadas del repositorio oficial
https://github.com/YiboZhao624/E-2GraphRAG (commit f598d4f0660b39ad3ad43f464386a80887d4ca5d).
Copyright (c) 2025 Yibo Zhao. Licencia MIT: ver licenses/E-2GraphRAG-LICENSE.

Cambios respecto al original, además de adaptar el código a este proyecto:
- Las llamadas al LLM son concurrentes contra vLLM y se miden con GPUTokenTracker.
- Las instrucciones dicen "conversation history" en vez de "novel".
- Solo se unen fragmentos contiguos entre hojas: el original interpreta como posición el
  nivel de los nodos de resumen.
- `_detect_neighbor_nodes` consulta el índice inverso con el identificador completo: el
  original usa una clave numérica que nunca coincide.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.data_loader import TokenPartition
from src.inference_setup import LocalInferenceEngine

SUMMARIZE_DETAILS_PROMPT = 'You are a helpful assistant that summarizes the details of a conversation history. You will be given a part of a conversation history. You need to summarize given content. The summary should include the main characters, the main plot and some other details. You need to return the summary in a concise manner without any additional fictive information. The length of the summary should be about 1000 tokens. \nHere is the content:\nContent: {content}\nNow, please summarize the content.\nSummary: '

SUMMARIZE_SUMMARY_PROMPT = 'You are a helpful assistant that further summarizes the summaries of a conversation history. You will be given a series of summaries of parts of a conversation history. You need to summarize the summaries in a concise manner. The length of the summary should be about 1000 tokens.\nHere is the summaries:\nSummary: {summary}\nNow, please summarize the summary based on the question.\nSummary: '

QA_PROMPT = 'You are a helpful assistant, you are given a question, please answer the question based on the given evidences. The answer should be a short sentence that supported by the given edidences and matches the requirements of the question. You should not assume any information beyond the evidence. You should only output the answer. The format of the Evidence is keyEntity1_keyEntity2: Related chunks, which means the related chunks contain the information of the relationship between keyEntity1 and keyEntity2, where keyEntity1 and keyEntity2 are the entities in the question.\n\nQuestion: {question}\nEvidence: {evidence}\n\nAnswer: '

# Parámetros de configs/example_config.yaml del repositorio oficial
SHORTEST_PATH_K = 4
MAX_CHUNKS = 25
SUMMARY_MAX_TOKENS = 1200
ANSWER_MAX_TOKENS = 300


@dataclass
class TreeNode:
    """Nodo del árbol de resúmenes: una hoja (`leaf_i`) o un resumen (`summary_nivel_i`)."""

    text: str
    children: List[str] = field(default_factory=list)


def extract_terms(doc: Any) -> Dict[str, Any]:
    """Extrae entidades y sustantivos de un Doc de spaCy, con su coocurrencia por oración."""
    cooccurrence: Dict[Tuple[str, str], int] = {}
    all_terms: Set[str] = set()
    appearance_count: Dict[str, int] = {}

    def _count(term: str) -> None:
        appearance_count[term] = appearance_count.get(term, 0) + 1

    for sent in doc.sents:
        sentence_terms: List[str] = []
        entity_positions = set()
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                # Un nombre como John Brown se separa en sus partes
                name_parts = ent.text.split()
                terms = name_parts if len(name_parts) >= 2 else [ent.text]
            elif ent.label_ in ("ORG", "GPE"):
                terms = [ent.text]
            else:
                terms = []
            sentence_terms.extend(terms)
            for term in terms:
                _count(term)
            entity_positions.update(token.i for token in ent)

        for token in sent:
            if token.i in entity_positions:
                continue
            if token.pos_ in ("NOUN", "PROPN") and token.lemma_.strip():
                term = token.lemma_.lower()
                sentence_terms.append(term)
                _count(term)

        all_terms.update(sentence_terms)
        for first, second in combinations(sentence_terms, 2):
            pair = tuple(sorted((first, second)))
            cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

    return {
        "terms": list(all_terms),
        "cooccurrence": cooccurrence,
        "appearance_count": appearance_count,
    }


class E2GraphRAGIndexer:
    """Índice de E²GraphRAG: grafo de coocurrencia en CPU y árbol de resúmenes en GPU."""

    def __init__(self, config: ExperimentConfig, engine: LocalInferenceEngine) -> None:
        self.config = config
        self.engine = engine
        self.graph = nx.Graph()
        self.tree: Dict[str, TreeNode] = {}
        self.index: Dict[str, List[str]] = {}
        self.inverse_index: Dict[str, List[str]] = {}
        self.appearance_count: Dict[str, Dict[str, int]] = {}
        self.node_ids: List[str] = []
        self.node_embeddings: Optional[np.ndarray] = None

    def _merge_sequential(self, texts: List[str], overlap: int) -> str:
        """Une fragmentos consecutivos quitando su solapamiento en tokens."""
        enc = self.engine._get_tokenizer()
        merged = texts[0]
        for text in texts[1:]:
            merged += enc.decode(enc.encode(text, disallowed_special=())[overlap:])
        return merged

    def _extract_cpu_graph(self, chunks: List[str]) -> Tuple[int, int]:
        """Construye en CPU el grafo, el índice invertido y los conteos de cada hoja."""
        nlp = self.engine.load_spacy_pipeline(lang="en")
        edge_weights: Dict[Tuple[str, str], int] = {}
        docs = nlp.pipe(chunks, batch_size=32)
        for chunk_idx, doc in enumerate(
            tqdm(docs, total=len(chunks), desc="E2GraphRAG [CPU]: grafo", leave=False)
        ):
            leaf_id = f"leaf_{chunk_idx}"
            extracted = extract_terms(doc)
            self.tree[leaf_id] = TreeNode(text=chunks[chunk_idx])
            self.appearance_count[leaf_id] = extracted["appearance_count"]
            self.inverse_index[leaf_id] = extracted["terms"]
            for term in extracted["terms"]:
                self.index.setdefault(term, []).append(leaf_id)
            for pair, weight in extracted["cooccurrence"].items():
                edge_weights[pair] = edge_weights.get(pair, 0) + weight

        for (first, second), weight in edge_weights.items():
            self.graph.add_edge(first, second, weight=weight)
        return self.graph.number_of_nodes(), self.graph.number_of_edges()

    async def _build_gpu_tree(self) -> int:
        """Resume de a `branching` nodos por nivel hasta que queden pocos resúmenes."""
        branching = self.config.e2_summary_tree_branching
        overlap = self.config.chunk_overlap_token_size
        semaphore = asyncio.Semaphore(min(32, self.config.max_num_seqs))

        async def _summarize(prompt: str) -> str:
            async with semaphore:
                return await self.engine.generate_async(
                    prompt=prompt, temperature=0.0, max_tokens=SUMMARY_MAX_TOKENS
                )

        async def _build_level(
            level: int, child_ids: List[str], template: str, slot: str, merge_overlap: int
        ) -> List[str]:
            groups = [child_ids[i : i + branching] for i in range(0, len(child_ids), branching)]
            prompts = [
                template.format(
                    **{slot: self._merge_sequential([self.tree[c].text for c in group], merge_overlap)}
                )
                for group in groups
            ]
            summaries = await asyncio.gather(*(_summarize(p) for p in prompts))
            level_ids = []
            for idx, (group, summary) in enumerate(zip(groups, summaries)):
                node_id = f"summary_{level}_{idx}"
                self.tree[node_id] = TreeNode(text=summary, children=group)
                level_ids.append(node_id)
            return level_ids

        current = await _build_level(
            0, list(self.tree), SUMMARIZE_DETAILS_PROMPT, "content", overlap
        )
        level = 1
        while len(current) > 1.2 * branching:
            current = await _build_level(level, current, SUMMARIZE_SUMMARY_PROMPT, "summary", 0)
            level += 1

        self.node_ids = list(self.tree)
        self.node_embeddings = self.engine.encode_texts(
            [self.tree[node_id].text for node_id in self.node_ids]
        )
        return len(self.tree)

    async def index_partition_async(self, partition: TokenPartition) -> Dict[str, Any]:
        """Ejecuta el pipeline completo sobre una partición y retorna sus métricas."""
        self.graph.clear()
        self.tree.clear()
        self.index.clear()
        self.inverse_index.clear()
        self.appearance_count.clear()
        self.engine.tracker.reset()

        t0_total = time.perf_counter()

        t0_cpu = time.perf_counter()
        chunks = [
            chunk
            for document in partition.documents
            for chunk in self.engine.split_into_chunks(
                document,
                self.config.e2_chunk_token_size,
                self.config.chunk_overlap_token_size,
            )
        ]
        num_nodes, num_edges = self._extract_cpu_graph(chunks)
        cpu_time_s = time.perf_counter() - t0_cpu

        t0_gpu = time.perf_counter()
        total_tree_nodes = await self._build_gpu_tree()
        gpu_time_s = time.perf_counter() - t0_gpu

        total_wall_s = time.perf_counter() - t0_total
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
            "latencia_s_por_mb": round(total_wall_s / partition.size_mb, 3),
            "gpu_llm_prompt_tokens": usage["prompt_tokens"],
            "gpu_llm_completion_tokens": usage["completion_tokens"],
            "gpu_llm_total_tokens": usage["total_gpu_llm_tokens"],
            "gpu_embedding_tokens": usage["embedding_tokens"],
            "llm_calls": usage["llm_calls"],
            "nodos_grafo": num_nodes,
            "aristas_grafo": num_edges,
            "nodos_arbol_resumen": total_tree_nodes,
            "llm_llamadas_fallidas": 0,
            "peak_vram_embeddings_gb": usage["peak_vram_gb"],
        }

    # -------------------------------------------------------------------------
    # Recuperación adaptativa
    # -------------------------------------------------------------------------

    def _dense_candidates(self, question: str, k: int) -> List[str]:
        question_vec = self.engine.encode_texts([question])[0]
        order = np.argsort(self.node_embeddings @ question_vec)[::-1][:k]
        return [self.node_ids[i] for i in order]

    def _local_retrieval(self, entities: List[str], k: int) -> Dict[str, List[str]]:
        """Fragmentos con ambas entidades de cada par a menos de `k` nodos en el grafo."""
        chunks_by_pair: Dict[str, List[str]] = {}
        for head, tail in combinations(entities, 2):
            if head not in self.graph or tail not in self.graph:
                continue
            try:
                path = nx.shortest_path(self.graph, head, tail)
            except nx.NetworkXNoPath:
                continue
            if len(path) <= k:
                shared = set(self.index.get(head, [])) & set(self.index.get(tail, []))
                chunks_by_pair[f"{head}_{tail}"] = sorted(shared)
        return self._merge_keys(chunks_by_pair)

    @staticmethod
    def _merge_keys(chunks_by_key: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Agrupa bajo una sola clave los fragmentos que aparecen en varias."""
        keys_by_chunk: Dict[str, Set[str]] = defaultdict(set)
        for key, chunks in chunks_by_key.items():
            for chunk in chunks:
                keys_by_chunk[chunk].add(key)
        merged: Dict[str, List[str]] = {}
        for chunk, keys in keys_by_chunk.items():
            if len(keys) > 1:
                entities = {entity for key in keys for entity in key.split("_")}
                new_key = "_".join(sorted(entities))
            else:
                new_key = next(iter(keys))
            merged.setdefault(new_key, []).append(chunk)
        return merged

    def _detect_neighbor_nodes(self, keys: Set[str], chunk_id: str) -> List[str]:
        """Hojas contiguas a `chunk_id` que también mencionan todas las entidades."""
        position = int(chunk_id.split("_")[-1])
        neighbors = [chunk_id]
        for step in (-1, 1):
            current = position + step
            while keys <= set(self.inverse_index.get(f"leaf_{current}", [])):
                neighbors.append(f"leaf_{current}")
                current += step
        return neighbors

    def _entity_aware_filter(
        self, candidates: Dict[str, List[str]], entities: List[str], top_k: int
    ) -> Dict[str, List[str]]:
        """Prioriza por entidades distintas, hojas contiguas y número de apariciones."""
        ranked = []
        for key, chunk_ids in candidates.items():
            key_entities = set(key.split("_"))
            for chunk_id in chunk_ids:
                ranked.append(
                    (
                        len(key_entities),
                        len(self._detect_neighbor_nodes(key_entities, chunk_id)),
                        sum(self.appearance_count[chunk_id].get(e, 0) for e in key_entities),
                        chunk_id,
                    )
                )
        ranked.sort(reverse=True)
        filtered: Dict[str, List[str]] = {}
        for *_, chunk_id in ranked[:top_k]:
            for entity in entities:
                if chunk_id in self.index.get(entity, []):
                    filtered.setdefault(entity, []).append(chunk_id)
        return self._merge_keys(filtered)

    def _count_occurrences(self, node_id: str, entities: List[str]) -> int:
        if node_id.startswith("leaf_"):
            stats = self.appearance_count.get(node_id, {})
            return sum(stats.get(entity, 0) for entity in entities)
        return sum(self._count_occurrences(child, entities) for child in self.tree[node_id].children)

    def _occurrence_ranking(
        self, candidates: List[str], entities: List[str], top_k: int
    ) -> Dict[str, List[str]]:
        """Reordena candidatos densos según cuántas veces mencionan las entidades."""
        counts = [self._count_occurrences(node_id, entities) for node_id in candidates]
        if not any(counts):
            return {"": candidates[:top_k]}
        ranked = sorted(zip(counts, candidates), key=lambda item: item[0], reverse=True)
        selected = [node_id for count, node_id in ranked if count > 0][:top_k]
        filtered: Dict[str, List[str]] = {}
        for node_id in selected:
            for entity in entities:
                if entity in self.inverse_index.get(node_id, []):
                    filtered.setdefault(entity, []).append(node_id)
        return self._merge_keys(filtered)

    def _format_evidence(self, retrieved: Dict[str, List[str]]) -> str:
        """Da el formato `entidadA_entidadB: fragmentos`, uniendo las hojas contiguas."""
        overlap = self.config.chunk_overlap_token_size
        lines = []
        for key, node_ids in retrieved.items():
            leaves = sorted(
                (n for n in node_ids if n.startswith("leaf_")),
                key=lambda n: int(n.split("_")[1]),
            )
            runs: List[List[str]] = []
            for leaf in leaves:
                position = int(leaf.split("_")[1])
                if runs and int(runs[-1][-1].split("_")[1]) + 1 == position:
                    runs[-1].append(leaf)
                else:
                    runs.append([leaf])
            for run in runs:
                text = self._merge_sequential([self.tree[n].text for n in run], overlap)
                lines.append(f"{key}: {text}")
            for node_id in node_ids:
                if not node_id.startswith("leaf_"):
                    lines.append(f"{key}: {self.tree[node_id].text}")
        return "\n".join(lines)

    def retrieve(self, question: str) -> Tuple[str, str]:
        """Devuelve la evidencia para la pregunta y qué camino de recuperación se usó."""
        nlp = self.engine.load_spacy_pipeline(lang="en")
        entities = extract_terms(nlp(question.split("\n")[0]))["terms"]

        if not entities:
            return self._format_evidence({"": self._dense_candidates(question, MAX_CHUNKS)}), "global"

        path_k = SHORTEST_PATH_K
        retrieved = self._local_retrieval(entities, path_k)
        count = sum(len(chunks) for chunks in retrieved.values())

        if count == 0:
            candidates = self._dense_candidates(question, MAX_CHUNKS * 2)
            ranked = self._occurrence_ranking(candidates, entities, MAX_CHUNKS)
            return self._format_evidence(ranked), "occurrence_rerank"

        previous = retrieved
        while count > MAX_CHUNKS:
            previous = retrieved
            path_k -= 1
            retrieved = self._local_retrieval(entities, path_k)
            count = sum(len(chunks) for chunks in retrieved.values())

        if count > 0:
            return self._format_evidence(retrieved), "local"
        filtered = self._entity_aware_filter(previous, entities, MAX_CHUNKS)
        return self._format_evidence(filtered), "entity_aware"

    async def answer(self, question: str) -> str:
        evidence, _ = self.retrieve(question)
        return await self.engine.generate_async(
            prompt=QA_PROMPT.format(question=question, evidence=evidence),
            temperature=0.0,
            max_tokens=ANSWER_MAX_TOKENS,
        )
