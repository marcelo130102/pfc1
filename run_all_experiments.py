"""
Orquestador unificado para ejecución desatendida / en segundo plano ($0 API Cost).
Permite lanzar el pipeline experimental completo del PFC de una sola vez:
1. Verificación de Hardware y Servidor vLLM en GPU (A100 / L4).
2. Experimento 1: Eficiencia en Indexación (E2GraphRAG vs LightRAG).
3. Experimento 2: Consistencia Bi-temporal (Zep/Graphiti vs LightRAG) + Juez LLM.
4. Generación de Figuras Publication-Ready (300 DPI).
5. Análisis estadístico y sincronización automática de CSVs y PDFs/PNGs a Google Drive.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# Asegurar que el directorio raíz esté en sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.analysis import run_analysis
from src.config import ExperimentConfig
from src.data_loader import DATASET_PROVENANCE
from src.exp1_indexing import run_experiment_1
from src.exp2_bitemporal import run_experiment_2
from src.inference_setup import LocalInferenceEngine
from src.visualization import generate_all_figures


def write_run_manifest(
    config: ExperimentConfig, hardware: str, started_at: str, status: str
) -> None:
    """Guarda qué código, modelos y datos produjeron los resultados de esta corrida."""
    code_files = [Path(BASE_DIR) / "run_all_experiments.py"] + sorted(
        (Path(BASE_DIR) / "src").glob("*.py")
    )
    packages = {}
    for name in ("vllm", "lightrag-hku", "graphiti-core", "openai", "spacy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    manifest = {
        "status": status,
        "pilot": config.pilot,
        "started_at_utc": started_at,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "llm_model": config.llm_model_id,
        "embedding_model": config.embedding_model_id,
        "graph_backend": f"neo4j {os.getenv('PFC_NEO4J_VERSION', 'desconocida')}",
        "exp1_token_partitions": config.exp1_token_partitions,
        "exp1_num_runs": config.exp1_num_runs,
        "dataset": dict(DATASET_PROVENANCE),
        "exp2_max_longmemeval_cases": config.exp2_max_longmemeval_cases,
        "exp2_synthetic_scenarios": config.exp2_num_synthetic_scenarios,
        "exp2_scenario_seed": config.exp2_scenario_seed,
        "resume_enabled": os.getenv("PFC_RESUME", "1") != "0",
        "python": platform.python_version(),
        "packages": packages,
        "code_sha256": {
            str(p.relative_to(BASE_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code_files
        },
    }
    out_path = config.base_dir / config.run_manifest_path
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    config.sync_to_google_drive(out_path)


async def main() -> None:
    t_start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    print("=" * 75)
    print("   PIPELINE EXPERIMENTAL COMPLETO DE GRAPHRAG (PFC) - EJECUCIÓN DESATENDIDA")
    print("=" * 75)

    # 1. Configuración de Hardware
    config = ExperimentConfig()
    perfil_hw = config.auto_configure_for_colab_gpu()
    print(f"\n[1/5] Perfil de Hardware Activo: {perfil_hw}")

    engine = LocalInferenceEngine(config)

    # 2. Inicialización de Servidor vLLM
    print("\n[2/5] Verificando / Levantando servidor vLLM en localhost:8000...")
    if not engine.is_server_healthy():
        engine.start_vllm_server(timeout_seconds=900)
    else:
        print("[OK] Servidor vLLM ya se encuentra activo y saludable.")
    print(f"[INFO] Modelo LLM activo: {config.llm_model_id}")
    write_run_manifest(config, perfil_hw, started_at, status="en_curso")

    # 3. Experimento 1: Indexación escalonada
    print("\n" + "=" * 75)
    print("[3/5] INICIANDO EXPERIMENTO 1: EFICIENCIA EN INDEXACIÓN")
    print("=" * 75)
    df_exp1 = await run_experiment_1(config=config, engine=engine)
    print(f"[OK] Experimento 1 completado con éxito ({len(df_exp1)} registros guardados).")

    # 4. Experimento 2: Consistencia Bi-temporal + Juez LLM
    print("\n" + "=" * 75)
    print("[4/5] INICIANDO EXPERIMENTO 2: RECUPERACIÓN BI-TEMPORAL Y EVALUACIÓN LLM")
    print("=" * 75)
    df_exp2_det, df_exp2_res = await run_experiment_2(config=config, engine=engine)
    print(f"[OK] Experimento 2 completado con éxito ({len(df_exp2_det)} diálogos evaluados).")

    # 5. Visualización
    print("\n" + "=" * 75)
    print("[5/5] GENERANDO FIGURAS ACADÉMICAS PUBLICATION-READY (300 DPI)")
    print("=" * 75)
    run_analysis(config=config)
    fig1, fig2 = generate_all_figures(config=config)
    print(f"[OK] Figuras generadas:")
    print(f"     - Figura 1: {fig1.resolve()}")
    print(f"     - Figura 2: {fig2.resolve()}")

    write_run_manifest(config, perfil_hw, started_at, status="completado")

    elapsed_min = (time.time() - t_start) / 60.0
    print("\n" + "=" * 75)
    print(f"PIPELINE EXPERIMENTAL FINALIZADO EXITOSAMENTE EN {elapsed_min:.2f} MINUTOS.")
    print("Los resultados se han guardado localmente y sincronizado con Google Drive (si está montado).")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
