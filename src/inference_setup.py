"""
BLOQUE 1: Setup del Entorno e Inferencia Local ($0 Costo de API).
Incluye:
- Lanzador y monitor de servidor vLLM / Ollama en localhost:8000.
- Rastreador de consumo de tokens de GPU y memoria VRAM (GPUTokenTracker).
- Modelo de embeddings densos BAAI/bge-m3 en GPU.
- Pipeline SpaCy en CPU para E²GraphRAG.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI, OpenAI
from tqdm.auto import tqdm

from src.config import ExperimentConfig


class LLMGenerationError(RuntimeError):
    """El servidor de inferencia no pudo generar una respuesta real."""


@dataclass
class GPUTokenTracker:
    """
    Contador thread-safe de tokens procesados por el motor LLM en GPU
    y monitor de uso pico de memoria VRAM durante cada experimento.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    embedding_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self) -> None:
        """Reinicia los contadores antes de iniciar una corrida experimental."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.llm_calls = 0
            self.embedding_tokens = 0
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def record_llm_usage(self, prompt_toks: int, completion_toks: int) -> None:
        """Registra los tokens consumidos en una llamada de generación LLM."""
        with self._lock:
            self.prompt_tokens += int(prompt_toks)
            self.completion_tokens += int(completion_toks)
            self.total_tokens += int(prompt_toks) + int(completion_toks)
            self.llm_calls += 1

    def record_embedding_usage(self, emb_toks: int) -> None:
        """Registra los tokens procesados por el modelo de embeddings BAAI/bge-m3."""
        with self._lock:
            self.embedding_tokens += int(emb_toks)

    def get_peak_vram_gb(self) -> float:
        """Obtiene el pico de memoria VRAM (en GB) registrado en la GPU actual."""
        try:
            import torch

            if torch.cuda.is_available():
                return round(
                    torch.cuda.max_memory_allocated(0) / (1024**3), 3
                )
        except Exception:
            pass
        return 0.0

    def snapshot(self) -> Dict[str, Any]:
        """Devuelve un resumen inmutable de las métricas acumuladas."""
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_gpu_llm_tokens": self.total_tokens,
                "llm_calls": self.llm_calls,
                "embedding_tokens": self.embedding_tokens,
                "peak_vram_gb": self.get_peak_vram_gb(),
            }


class LocalInferenceEngine:
    """
    Gestor unificado del motor de inferencia local (vLLM u Ollama) y del modelo
    de embeddings BAAI/bge-m3 en Google Colab Pro.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.tracker = GPUTokenTracker()
        self.vllm_process: Optional[subprocess.Popen] = None
        self._embedder: Any = None
        self._spacy_nlp: Any = None
        self._tokenizer: Any = None
        self._async_client: Optional[AsyncOpenAI] = None
        self._sync_client: Optional[OpenAI] = None
        # LightRAG captura internamente los errores del LLM: este contador permite
        # detectar los fallos aunque la excepción no se propague hasta el experimento.
        self.llm_failure_count: int = 0
        self.last_llm_error: Optional[Exception] = None

    def _get_tokenizer(self):
        """Inicializa un tokenizador rápido (tiktoken cl100k_base) para conteo de tokens."""
        if self._tokenizer is None:
            try:
                import tiktoken

                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Cuenta el número exacto de tokens de un texto."""
        if not text:
            return 0
        enc = self._get_tokenizer()
        if enc is not None:
            return len(enc.encode(text, disallowed_special=()))
        # Fallback aproximado por palabras si tiktoken no está disponible
        return max(1, int(len(text.split()) * 1.3))

    def split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Divide un texto en fragmentos de `chunk_size` tokens con solapamiento."""
        enc = self._get_tokenizer()
        if enc is None:
            raise RuntimeError("tiktoken es necesario para fragmentar por tokens.")
        tokens = enc.encode(text, disallowed_special=())
        step = chunk_size - overlap
        return [
            enc.decode(tokens[start : start + chunk_size])
            for start in range(0, max(len(tokens) - overlap, 1), step)
        ]

    def is_server_healthy(self) -> bool:
        """
        Verifica si el servidor OpenAI-compatible responde en localhost:8000 y
        sincroniza automáticamente `self.config.llm_model_id` con el modelo activo en vLLM.
        """
        import json

        url = f"{self.config.base_url}/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    models_list = payload.get("data", [])
                    if models_list and models_list[0].get("id"):
                        active_id = models_list[0]["id"]
                        if self.config.llm_model_id != active_id:
                            self.config.llm_model_id = active_id
                    return True
                return False
        except Exception:
            return False

    def start_vllm_server(self, timeout_seconds: int = 300) -> bool:
        """
        Inicia el servidor vLLM en segundo plano con optimizaciones para GPU A100/L4
        (FlashAttention, Prefix Caching y bfloat16). Si no encuentra el modelo
        principal (ej. falta HF_TOKEN para Llama-3.1), conmuta a Qwen2.5-7B-Instruct.
        """
        if self.is_server_healthy():
            print(f"[INFO] Servidor de inferencia ya activo en {self.config.base_url}")
            return True

        model_to_load = self.config.llm_model_id
        hf_token = os.getenv("HF_TOKEN", "")
        if "meta-llama" in model_to_load and not hf_token:
            print(
                "[AVISO] No se detectó HF_TOKEN en el entorno para meta-llama. "
                f"Conmutando automáticamente al modelo abierto: {self.config.fallback_llm_model_id}"
            )
            model_to_load = self.config.fallback_llm_model_id
            self.config.llm_model_id = model_to_load

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_to_load,
            "--host",
            self.config.vllm_host,
            "--port",
            str(self.config.vllm_port),
            "--dtype",
            self.config.dtype,
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
            "--max-model-len",
            str(self.config.max_model_len),
            "--max-num-seqs",
            str(self.config.max_num_seqs),
            "--enable-prefix-caching",
            "--disable-log-requests",
        ]

        log_path = self.config.base_dir / "vllm_server.log"
        print(f"[INFO] Lanzando servidor vLLM con {model_to_load}...")
        print(f"[INFO] Logs del servidor en: {log_path}")

        try:
            log_file = open(log_path, "w", encoding="utf-8")
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )

            with tqdm(
                total=timeout_seconds,
                desc="Esperando inicialización de vLLM en GPU",
                unit="s",
            ) as pbar:
                elapsed = 0
                while elapsed < timeout_seconds:
                    if self.is_server_healthy():
                        print(
                            f"\n[OK] Servidor vLLM listo en {self.config.base_url} "
                            f"({elapsed}s)"
                        )
                        return True
                    if self.vllm_process.poll() is not None:
                        print(
                            "\n[ERROR] El proceso vLLM terminó prematuramente. "
                            "Revisa vllm_server.log."
                        )
                        break
                    time.sleep(2)
                    elapsed += 2
                    pbar.update(2)
        except Exception as exc:
            print(f"[ERROR] No se pudo iniciar vLLM: {exc}")

        raise RuntimeError(
            f"El servidor vLLM no quedó disponible en {self.config.base_url}. "
            f"Revisa {log_path}."
        )

    def load_embedding_model(self) -> Any:
        """Carga el modelo denso multilingüe BAAI/bge-m3 en GPU."""
        if self._embedder is not None:
            return self._embedder

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(
                f"[INFO] Cargando modelo de embeddings {self.config.embedding_model_id} "
                f"en dispositivo '{device}'..."
            )
            self._embedder = SentenceTransformer(
                self.config.embedding_model_id,
                device=device,
                model_kwargs=(
                    {"torch_dtype": torch.float16} if device == "cuda" else {}
                ),
            )
            print("[OK] Modelo BAAI/bge-m3 cargado correctamente.")
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar {self.config.embedding_model_id}: {exc}"
            ) from exc

        return self._embedder

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings normalizados (dim=1024) con BAAI/bge-m3 y contabiliza
        los tokens procesados en GPU.
        """
        if not texts:
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)

        total_toks = sum(self.count_tokens(t) for t in texts)
        self.tracker.record_embedding_usage(total_toks)

        embedder = self.load_embedding_model()
        vectors = embedder.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def load_spacy_pipeline(self, lang: str = "en") -> Any:
        """Carga el modelo SpaCy completo: E²GraphRAG usa sus etiquetas, lemas y oraciones."""
        if self._spacy_nlp is not None:
            return self._spacy_nlp

        import spacy

        model_name = (
            self.config.spacy_model_en
            if lang == "en"
            else self.config.spacy_model_es
        )
        nlp = spacy.load(model_name)
        # Aumentar límite de caracteres para los fragmentos más largos
        nlp.max_length = 10_000_000
        self._spacy_nlp = nlp
        print(f"[OK] Pipeline SpaCy '{model_name}' activo en CPU (pipes={nlp.pipe_names}).")
        return self._spacy_nlp

    def _llm_failure(self, exc: Exception) -> LLMGenerationError:
        """Registra un fallo de generación y construye la excepción a propagar."""
        self.llm_failure_count += 1
        self.last_llm_error = exc
        return LLMGenerationError(
            f"Fallo del LLM '{self.config.llm_model_id}' en {self.config.base_url}: "
            f"{exc.__class__.__name__}: {exc}"
        )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Realiza una llamada asíncrona al servidor local vLLM / Ollama y registra
        automáticamente los tokens de prompt y completion en GPUTokenTracker.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if self._async_client is None:
                self._async_client = AsyncOpenAI(
                    base_url=self.config.base_url, api_key=self.config.api_key
                )
            extra_args = {"response_format": response_format} if response_format else {}
            response = await self._async_client.chat.completions.create(
                model=self.config.llm_model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_args,
            )
            content = response.choices[0].message.content or ""
            if response.usage:
                self.tracker.record_llm_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
            else:
                p_toks = self.count_tokens(
                    (system_prompt or "") + "\n" + prompt
                )
                c_toks = self.count_tokens(content)
                self.tracker.record_llm_usage(p_toks, c_toks)
            return content.strip()
        except Exception as exc:
            raise self._llm_failure(exc) from exc

    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Versión síncrona de generación contra vLLM / Ollama con registro de tokens."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            client = OpenAI(
                base_url=self.config.base_url, api_key=self.config.api_key
            )
            response = client.chat.completions.create(
                model=self.config.llm_model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            if response.usage:
                self.tracker.record_llm_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
            else:
                p_toks = self.count_tokens(
                    (system_prompt or "") + "\n" + prompt
                )
                c_toks = self.count_tokens(content)
                self.tracker.record_llm_usage(p_toks, c_toks)
            return content.strip()
        except Exception as exc:
            raise self._llm_failure(exc) from exc
