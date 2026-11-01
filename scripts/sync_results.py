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

FILES_TO_SYNC = [
    "experimento1_resultados.csv",
    "experimento2_resultados.csv",
    "fig1_latencia_indexacion.png",
    "fig1_latencia_indexacion.pdf",
    "fig2_bitemporal_recall_errores.png",
    "fig2_bitemporal_recall_errores.pdf",
    "experiments.log",
]


def sync_from_colab(session: str = "pfc", remote_base: str = "/content/pfc1") -> None:
    print(f"=== Sincronizando resultados desde la sesión '{session}' de Colab ===")
    local_dir = Path(__file__).resolve().parent.parent

    for filename in FILES_TO_SYNC:
        remote_path = f"{remote_base}/{filename}"
        local_path = local_dir / filename
        cmd = ["colab", "download", "-s", session, remote_path, str(local_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                size_kb = local_path.stat().st_size / 1024 if local_path.exists() else 0
                print(f" [OK] {filename} descargado con éxito ({size_kb:.1f} KB)")
            else:
                # El archivo puede no haberse generado aún si el experimento sigue en curso
                print(f" [PENDIENTE] {filename} (aún no generado en la VM)")
        except Exception as exc:
            print(f" [ERROR] No se pudo descargar {filename}: {exc}")

    print("\nSincronización finalizada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sincronizar resultados de Colab a local.")
    parser.add_argument("--session", "-s", default="pfc", help="Nombre de la sesión de Colab")
    args = parser.parse_args()
    sync_from_colab(session=args.session)
