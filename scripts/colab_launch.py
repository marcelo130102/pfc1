"""
Lanzador dual para Opción 2 (Colab CLI desde Mac + Ejecución en VM de Colab).

Uso desde tu Mac:
    # 1. Crear sesión con GPU A100 y High-RAM y montar Drive
    colab new -s pfc --gpu A100 --high-mem
    colab drivemount -s pfc

    # Verificar LLM, LightRAG y dataset antes de gastar GPU en todo el pipeline:
    python3 scripts/colab_smoke_test.py -s pfc

    # 2A. Ampliar experimentos (conserva 100k-1M en Exp1, calcula 1.5M y 2.5M + 100 casos de 5 saltos en Exp2):
    python3 scripts/colab_launch.py -s pfc --expand

    # 2B. Recalcular absolutamente todo desde cero (100k a 2.5M + 100 casos de Exp2):
    python3 scripts/colab_launch.py -s pfc --fresh
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/marcelo130102/pfc1.git"
REPO_DIR = "/content/pfc1"
DRIVE_DIR = "/content/drive/MyDrive/PFC_RAG_Resultados"
POLL_SECONDS = 30
REPORT_EVERY_SECONDS = 120

PACKAGES = [
    "lightrag-hku", "spacy", "vllm", "graphiti-core", "datasets", "pandas",
    "networkx", "sentence-transformers", "FlagEmbedding", "tiktoken", "openai",
    "matplotlib", "seaborn", "tqdm", "nest_asyncio", "psutil",
]


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def running_pid():
    """Devuelve el PID de run_all_experiments.py si ya está corriendo en la VM."""
    out = subprocess.run(
        ["pgrep", "-f", "run_all_experiments.py"], capture_output=True, text=True
    ).stdout.split()
    return int(out[0]) if out else None


def tail(path, n=5):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:]).rstrip()
    except OSError:
        return "(log aún vacío)"


def prepare_repo_on_colab_vm() -> None:
    """Clona o actualiza el repositorio base en /content/pfc1 dentro de la VM."""
    if os.path.exists(REPO_DIR):
        subprocess.run(["git", "-C", REPO_DIR, "pull"], check=False)
    else:
        run(["git", "clone", REPO_URL, REPO_DIR])
    os.makedirs(os.path.join(REPO_DIR, "src"), exist_ok=True)


def start_on_colab_vm() -> None:
    """
    Se ejecuta DENTRO de la VM de Colab:
    Instala dependencias, gestiona los checkpoints según --fresh o --expand,
    y lanza `run_all_experiments.py` en segundo plano.
    """
    prepare_repo_on_colab_vm()

    drive_ok = os.path.isdir("/content/drive/MyDrive")
    if not drive_ok:
        print("[AVISO] Google Drive NO está montado. Ejecuta antes: colab drivemount -s pfc", flush=True)
    log_dir = DRIVE_DIR if drive_ok else REPO_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "experiments.log")

    mode_fresh = os.environ.get("PFC_FRESH") == "1"
    mode_expand = os.environ.get("PFC_EXPAND") == "1"

    pid = running_pid()
    if pid and (mode_fresh or mode_expand):
        print(f"[INFO] Deteniendo corrida anterior (PID {pid})...", flush=True)
        subprocess.run(["kill", "-9", str(pid)], check=False)
        time.sleep(2)
        pid = None

    if pid:
        print(f"[INFO] El pipeline ya estaba corriendo en la VM (PID {pid}).", flush=True)
    else:
        run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
        run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])
        run([sys.executable, "-m", "spacy", "download", "es_core_news_sm", "-q"])

        env = os.environ.copy()
        if mode_fresh:
            env["PFC_RESUME"] = "0"
            for old_csv in ("experimento1_resultados.csv", "experimento2_resultados.csv"):
                for folder in (DRIVE_DIR, REPO_DIR):
                    p = os.path.join(folder, old_csv)
                    if os.path.exists(p):
                        os.remove(p)
                        print(f"[FRESH] Checkpoint previo eliminado: {p}", flush=True)
            if os.path.exists(log_path):
                os.remove(log_path)
        elif mode_expand:
            # En modo --expand conservamos experimento1_resultados.csv (para reutilizar 100k..1M
            # y calcular sólo 1.5M y 2.5M) y limpiamos experimento2_resultados.csv para correr
            # los 100 casos ampliados de 5 saltos temporales desde cero.
            env["PFC_RESUME"] = "1"
            for folder in (DRIVE_DIR, REPO_DIR):
                p2 = os.path.join(folder, "experimento2_resultados.csv")
                if os.path.exists(p2):
                    os.remove(p2)
                    print(f"[EXPAND] Reiniciando Experimento 2 a 100 casos de 5 saltos: eliminado {p2}", flush=True)

        log_fh = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", "run_all_experiments.py"],
            cwd=REPO_DIR,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        pid = proc.pid
        print(f"[OK] Pipeline lanzado en la VM (PID {pid}). Log: {log_path}", flush=True)

    if os.environ.get("PFC_MAC_MANAGED") == "1":
        return

    t0 = time.time()
    last_report = 0.0
    while os.path.exists(f"/proc/{pid}"):
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        if time.time() - last_report >= REPORT_EVERY_SECONDS:
            mins = (time.time() - t0) / 60
            print(f"\n[{mins:6.1f} min] Último avance:\n{tail(log_path)}", flush=True)
            last_report = time.time()
        time.sleep(POLL_SECONDS)

    repo_log_path = os.path.join(REPO_DIR, "experiments.log")
    if log_path != repo_log_path and os.path.exists(log_path):
        shutil.copy2(log_path, repo_log_path)

    print("\n[FIN] El pipeline terminó en la VM. Últimas líneas del log:", flush=True)
    print(tail(log_path, n=15), flush=True)


REMOTE_POLL_SNIPPET = """
import os, shutil, subprocess
out = subprocess.run(["pgrep", "-f", "run_all_experiments.py"], capture_output=True, text=True).stdout.split()
log_p = "/content/drive/MyDrive/PFC_RAG_Resultados/experiments.log"
if not os.path.exists(log_p):
    log_p = "/content/pfc1/experiments.log"
lines = ""
if os.path.exists(log_p):
    with open(log_p, "r", encoding="utf-8", errors="replace") as f:
        lines = "".join(f.readlines()[-6:]).rstrip()
if out:
    print("STATUS:RUNNING:" + out[0])
    print(lines)
else:
    if os.path.exists(log_p) and log_p != "/content/pfc1/experiments.log":
        shutil.copy2(log_p, "/content/pfc1/experiments.log")
    print("STATUS:DONE")
    print(lines)
"""


def sync_local_code_to_vm(session: str, repo_root: Path) -> None:
    """
    Sube los módulos locales (`src/*.py`, `run_all_experiments.py` y `experimento1_resultados.csv`
    si existe) desde tu Mac hacia `/content/pfc1` en la VM para garantizar que corra siempre
    la última versión ampliada sin depender de `git push`.
    """
    init_snippet = (
        "import os, subprocess\n"
        f"if not os.path.exists('{REPO_DIR}'):\n"
        f"    subprocess.run(['git', 'clone', '{REPO_URL}', '{REPO_DIR}'], check=False)\n"
        f"os.makedirs('{REPO_DIR}/src', exist_ok=True)\n"
        f"os.makedirs('{REPO_DIR}/scripts', exist_ok=True)\n"
    )
    init_res = subprocess.run(
        ["colab", "exec", "-s", session],
        input=init_snippet,
        text=True,
        capture_output=True,
        cwd=str(repo_root),
    )
    if init_res.returncode != 0:
        raise RuntimeError(
            f"No se pudo preparar {REPO_DIR} en la VM: {init_res.stderr.strip()}"
        )

    files_to_upload = [
        (repo_root / "run_all_experiments.py", f"{REPO_DIR}/run_all_experiments.py"),
        (repo_root / "src" / "__init__.py", f"{REPO_DIR}/src/__init__.py"),
        (repo_root / "src" / "config.py", f"{REPO_DIR}/src/config.py"),
        (repo_root / "src" / "inference_setup.py", f"{REPO_DIR}/src/inference_setup.py"),
        (repo_root / "src" / "data_loader.py", f"{REPO_DIR}/src/data_loader.py"),
        (repo_root / "src" / "exp1_indexing.py", f"{REPO_DIR}/src/exp1_indexing.py"),
        (repo_root / "src" / "exp2_bitemporal.py", f"{REPO_DIR}/src/exp2_bitemporal.py"),
        (repo_root / "src" / "visualization.py", f"{REPO_DIR}/src/visualization.py"),
        (repo_root / "scripts" / "colab_launch.py", f"{REPO_DIR}/scripts/colab_launch.py"),
        (repo_root / "scripts" / "colab_smoke_test.py", f"{REPO_DIR}/scripts/colab_smoke_test.py"),
    ]
    exp1_local = repo_root / "experimento1_resultados.csv"
    if exp1_local.exists():
        files_to_upload.append((exp1_local, f"{REPO_DIR}/experimento1_resultados.csv"))

    print("[SYNC] Subiendo código ampliado desde tu Mac a la VM de Colab...", flush=True)
    for local_f, remote_f in files_to_upload:
        res = subprocess.run(
            ["colab", "upload", "-s", session, str(local_f), remote_f],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"No se pudo subir {local_f} a la VM: {res.stderr.strip()}"
            )


def run_from_mac_laptop() -> None:
    """
    Se ejecuta en tu Mac (macOS):
    1. Activa `caffeinate -dims` ligado al PID del script.
    2. Sincroniza el código local ampliado hacia `/content/pfc1` en la VM.
    3. Lanza el trabajo en la VM de Colab y monitorea cada 30s sin TimeoutError.
    4. Al terminar, descarga automáticamente todos los CSVs, figuras y logs a tu Mac.
    """
    parser = argparse.ArgumentParser(
        description="Lanza los experimentos en Colab con caffeinate y descarga los resultados al terminar."
    )
    parser.add_argument("--session", "-s", default="pfc", help="Nombre de la sesión de Colab (default: pfc)")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Conserva 100k-1M de Exp1, calcula las nuevas particiones 1.5M y 2.5M, y corre los 100 casos de 5 saltos de Exp2.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Elimina todos los checkpoints previos y recalcula de 100k a 2.5M + 100 casos de Exp2 desde cero.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.sync_results import sync_from_colab

    caffeinate_proc = None
    try:
        caffeinate_proc = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
        print(
            f"[CAFFEINATE] Activo en tu Mac (PID {caffeinate_proc.pid}) -> "
            "Tu laptop NO entrará en reposo mientras dure la ejecución.",
            flush=True,
        )
    except FileNotFoundError:
        print("[AVISO] No se encontró el binario 'caffeinate' en el sistema.", flush=True)

    try:
        sync_local_code_to_vm(session=args.session, repo_root=repo_root)

        header = (
            "import os\n"
            "os.environ['PFC_MAC_MANAGED'] = '1'\n"
            f"os.environ['PFC_FRESH'] = '{'1' if args.fresh else '0'}'\n"
            f"os.environ['PFC_EXPAND'] = '{'1' if args.expand else '0'}'\n"
        )
        full_code = header + script_path.read_text(encoding="utf-8")

        print(f"[COLAB] Inicializando pipeline ampliado en la sesión '{args.session}'...", flush=True)
        subprocess.run(
            ["colab", "exec", "-s", args.session],
            input=full_code,
            text=True,
            cwd=str(repo_root),
            check=False,
        )

        # Bucle de monitoreo mediante llamadas cortas (mantiene viva la VM sin TimeoutError)
        t0 = time.time()
        last_report = 0.0
        while True:
            time.sleep(POLL_SECONDS)
            res = subprocess.run(
                ["colab", "exec", "-s", args.session],
                input=REMOTE_POLL_SNIPPET,
                text=True,
                capture_output=True,
                cwd=str(repo_root),
            )
            out_text = (res.stdout or "").strip()
            if "STATUS:DONE" in out_text:
                tail_text = out_text.split("STATUS:DONE", 1)[-1].strip()
                print(f"\n[FIN] El pipeline terminó en la VM. Últimas líneas del log:\n{tail_text}", flush=True)
                break

            if time.time() - last_report >= REPORT_EVERY_SECONDS:
                mins = (time.time() - t0) / 60.0
                tail_text = "\n".join(out_text.splitlines()[1:]).strip()
                print(f"\n[{mins:5.1f} min] Progreso en VM:\n{tail_text}", flush=True)
                last_report = time.time()

    except KeyboardInterrupt:
        print("\n[AVISO] Interrupción manual en Mac. Sincronizando avances actuales...", flush=True)
    finally:
        sync_from_colab(session=args.session)
        if caffeinate_proc and caffeinate_proc.poll() is None:
            caffeinate_proc.terminate()
        print(
            f"Recuerda liberar la VM cuando verifiques tus archivos con: colab stop -s {args.session}",
            flush=True,
        )


if __name__ == "__main__":
    if sys.platform == "darwin":
        run_from_mac_laptop()
    else:
        start_on_colab_vm()
