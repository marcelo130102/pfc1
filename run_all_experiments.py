"""
Orquestador unificado para ejecución desatendida / en segundo plano ($0 API Cost).
Permite lanzar el pipeline experimental completo del PFC de una sola vez:
1. Verificación de Hardware y Servidor vLLM en GPU (A100 / L4).
2. Experimento 1: Eficiencia en Indexación (E2GraphRAG vs LightRAG).
3. Experimento 2: Consistencia Bi-temporal (Zep/Graphiti vs LightRAG) + Juez LLM.
4. Generación de Figuras Publication-Ready (300 DPI).
5. Sincronización automática de CSVs y PDFs/PNGs a Google Drive.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

# Asegurar que el directorio raíz esté en sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import ExperimentConfig
from src.exp1_indexing import run_experiment_1
from src.exp2_bitemporal import run_experiment_2
from src.inference_setup import LocalInferenceEngine
from src.visualization import generate_all_figures


async def main() -> None:
    t_start = time.time()
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
        engine.start_vllm_server(timeout_seconds=300)
    else:
        print("[OK] Servidor vLLM ya se encuentra activo y saludable.")

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
    fig1, fig2 = generate_all_figures(config=config)
    print(f"[OK] Figuras generadas:")
    print(f"     - Figura 1: {fig1.resolve()}")
    print(f"     - Figura 2: {fig2.resolve()}")

    elapsed_min = (time.time() - t_start) / 60.0
    print("\n" + "=" * 75)
    print(f"PIPELINE EXPERIMENTAL FINALIZADO EXITOSAMENTE EN {elapsed_min:.2f} MINUTOS.")
    print("Los resultados se han guardado localmente y sincronizado con Google Drive (si está montado).")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
