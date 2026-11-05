"""
Descarga automáticamente todos los resultados generados desde la sesión de Colab
a este repositorio local en tu Mac.

Uso:
    python scripts/sync_results.py
    # O especificando una sesión distinta:
    python scripts/sync_results.py --session pfc
"""

import argparse
import subprocess
from pathlib import Path

RESULT_FILES = [
    "experimento1_resultados.csv",
    "experimento1_calidad.csv",
    "experimento2_resultados.csv",
    "resumen_estadistico.csv",
    "muestra_revision_juez.csv",
    "fig1_latencia_indexacion.png",
    "fig1_latencia_indexacion.pdf",
    "fig2_bitemporal_recall_errores.png",
    "fig2_bitemporal_recall_errores.pdf",
    "run_manifest.json",
]
FILES_TO_SYNC = (
    RESULT_FILES
    + [f"piloto_{name}" for name in RESULT_FILES]
    + ["experiments.log", "vllm_server.log"]
)

REMOTE_DIRS = [
    "/content/pfc1",
    "/content/drive/MyDrive/PFC_RAG_Resultados",
]


def sync_from_colab(session: str = "pfc", remote_base: str = "/content/pfc1") -> None:
    print(f"\n=== Descargando resultados desde la sesión '{session}' de Colab a tu Mac ===")
    local_dir = Path(__file__).resolve().parent.parent
    search_dirs = [remote_base] + [d for d in REMOTE_DIRS if d != remote_base]

    for filename in FILES_TO_SYNC:
        local_path = local_dir / filename
        downloaded = False

        for r_dir in search_dirs:
            remote_path = f"{r_dir}/{filename}"
            cmd = ["colab", "download", "-s", session, remote_path, str(local_path)]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and local_path.exists() and local_path.stat().st_size > 0:
                    size_kb = local_path.stat().st_size / 1024
                    print(f" [OK] {filename} descargado en tu laptop ({size_kb:.1f} KB)")
                    downloaded = True
                    break
            except Exception as exc:
                print(f" [ERROR] Fallo al invocar colab download para {filename}: {exc}")
                break

        if not downloaded:
            print(f" [PENDIENTE] {filename} (aún no generado en la VM)")

    print(f"\n[OK] Sincronización finalizada en: {local_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sincronizar resultados de Colab a local.")
    parser.add_argument("--session", "-s", default="pfc", help="Nombre de la sesión de Colab")
    args = parser.parse_args()
    sync_from_colab(session=args.session)
