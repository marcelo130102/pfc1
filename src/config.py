"""
Configuración centralizada del proyecto experimental RAG (Python 3.10+).
Detecta automáticamente el perfil de hardware en Google Colab Pro (A100, L4, T4 o CPU)
y ajusta los hiperparámetros de vLLM, concurrencia y procesamiento multiproceso en CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    """Parámetros globales para los Bloques 1, 2, 3 y 4 en Google Colab Pro."""

    # --- Modelos de Inferencia Local ($0 API Cost) ---
    llm_model_id: str = os.getenv(
        "PFC_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )
    fallback_llm_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    embedding_model_id: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # --- Endpoint OpenAI-Compatible Local (vLLM / Ollama) ---
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    api_key: str = "EMPTY"

    # --- Modelos SpaCy para extracción NER en CPU (E2GraphRAG) ---
    spacy_model_en: str = "en_core_web_sm"
    spacy_model_es: str = "es_core_news_sm"

    # --- Parámetros de Hardware (Auto-ajustables según GPU detectada) ---
    gpu_memory_utilization: float = 0.80
    max_model_len: int = 32768
    max_num_seqs: int = 64
    embedding_batch_size: int = 64
    spacy_n_process: int = 4
    dtype: str = "bfloat16"

    # --- Experimento 1: Particiones escalonadas de LongMemEval (en tokens) ---
    exp1_token_partitions: List[int] = field(
        default_factory=lambda: [100_000, 250_000, 500_000, 1_000_000]
    )
    e2_chunk_token_size: int = 1200
    e2_summary_tree_branching: int = 5
    e2_summary_max_levels: int = 3

    # --- Experimento 2: Evaluación Bi-temporal en LongMemEval ---
    exp2_num_dialogues: int = 30
    exp2_top_k_retrieval: int = 5

    # --- Rutas de Salida y Persistencia ---
    base_dir: Path = Path(os.getenv("PFC_BASE_DIR", "."))
    exp1_csv_path: str = "experimento1_resultados.csv"
    exp2_csv_path: str = "experimento2_resultados.csv"
    fig1_path: str = "fig1_latencia_indexacion.png"
    fig2_path: str = "fig2_bitemporal_recall_errores.png"

    @property
    def base_url(self) -> str:
        """Retorna la URL base compatible con el cliente de OpenAI."""
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"

    def auto_configure_for_colab_gpu(self) -> str:
        """
        Inspecciona la GPU activa en Google Colab Pro y optimiza los parámetros
        para A100 (40GB/80GB), L4 (24GB) o entornos de prueba.
        """
        try:
            import multiprocessing
            import torch

            cpu_count = multiprocessing.cpu_count()
            self.spacy_n_process = max(1, min(8, cpu_count - 2))

            if not torch.cuda.is_available():
                self.dtype = "float32"
                return "CPU (Sin acelerador GPU detectado)"

            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            if "A100" in gpu_name or vram_gb >= 38.0:
                # Perfil Máximo: Google Colab Pro / AI Pro con NVIDIA A100
                self.gpu_memory_utilization = 0.80
                self.max_model_len = 32768
                self.max_num_seqs = 64
                self.embedding_batch_size = 128
                self.dtype = "bfloat16"
            elif "L4" in gpu_name or vram_gb >= 20.0:
                # Perfil Intermedio-Alto: NVIDIA L4 (24 GB VRAM)
                self.gpu_memory_utilization = 0.72
                self.max_model_len = 16384
                self.max_num_seqs = 24
                self.embedding_batch_size = 64
                self.dtype = "bfloat16"
            else:
                # Perfil T4 (16 GB VRAM)
                self.gpu_memory_utilization = 0.68
                self.max_model_len = 8192
                self.max_num_seqs = 8
                self.embedding_batch_size = 32
                self.dtype = "float16"

            return f"{gpu_name} ({vram_gb:.1f} GB VRAM) | {cpu_count} vCPUs"
        except Exception as exc:
            return f"Configuración por defecto (Detalle: {exc})"
