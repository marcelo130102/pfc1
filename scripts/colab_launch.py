"""
Lanzador para Colab CLI: prepara la VM y ejecuta el pipeline completo.

Uso desde tu Mac (la VM debe existir y tener Drive montado):
    colab new -s pfc --gpu A100 --high-mem
    colab drivemount -s pfc
    colab exec -s pfc -f scripts/colab_launch.py

Qué hace:
1. Clona o actualiza el repositorio en /content/pfc1.
2. Instala dependencias y modelos SpaCy.
3. Lanza run_all_experiments.py como proceso independiente (sobrevive aunque
   se corte la conexión con tu Mac) y escribe el log en Google Drive.
4. Se queda esperando hasta que el pipeline termine, imprimiendo el avance.
   Así el kernel de Colab permanece "activo" y la VM no se recicla por
   inactividad mientras el experimento corre.

Si la VM se pierde a mitad de camino, basta con volver a ejecutar estos mismos
comandos: los experimentos se reanudan desde los CSV guardados en Drive.
"""

import os
import subprocess
import sys
import time

REPO_URL = "https://github.com/marcelo130102/pfc1.git"
REPO_DIR = "/content/pfc1"
DRIVE_DIR = "/content/drive/MyDrive/PFC_RAG_Resultados"
POLL_SECONDS = 60
REPORT_EVERY_SECONDS = 600

PACKAGES = [
    "lightrag-hku", "spacy", "vllm", "graphiti-core", "datasets", "pandas",
    "networkx", "sentence-transformers", "FlagEmbedding", "tiktoken", "openai",
    "matplotlib", "seaborn", "tqdm", "nest_asyncio", "psutil",
]


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def running_pid():
    """Devuelve el PID de run_all_experiments.py si ya está corriendo."""
    out = subprocess.run(
        ["pgrep", "-f", "run_all_experiments.py"], capture_output=True, text=True
    ).stdout.split()
    return int(out[0]) if out else None


def tail(path, n=3):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:]).rstrip()
    except OSError:
        return "(log aún vacío)"


# 1. Repositorio
if os.path.exists(REPO_DIR):
    run(["git", "-C", REPO_DIR, "pull"])
else:
    run(["git", "clone", REPO_URL, REPO_DIR])

# 2. Log en Drive (si está montado) para poder revisarlo aunque la VM muera
drive_ok = os.path.isdir("/content/drive/MyDrive")
if not drive_ok:
    print("[AVISO] Google Drive NO está montado. Ejecuta antes: colab drivemount -s pfc", flush=True)
log_dir = DRIVE_DIR if drive_ok else REPO_DIR
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "experiments.log")

pid = running_pid()
if pid:
    print(f"[INFO] El pipeline ya estaba corriendo (PID {pid}). Me engancho a esperar.", flush=True)
else:
    # 3. Dependencias
    run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])
    run([sys.executable, "-m", "spacy", "download", "es_core_news_sm", "-q"])

    # 4. Lanzar el pipeline en una sesión propia (independiente del kernel)
    log_fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", "run_all_experiments.py"],
        cwd=REPO_DIR,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid = proc.pid
    print(f"[OK] Pipeline lanzado (PID {pid}). Log: {log_path}", flush=True)

# 5. Mantener el kernel ocupado hasta que termine
t0 = time.time()
last_report = 0.0
while os.path.exists(f"/proc/{pid}"):
    try:
        # Recoge el proceso si terminó (evita zombies que mantendrían /proc/<pid>)
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    if time.time() - last_report >= REPORT_EVERY_SECONDS:
        mins = (time.time() - t0) / 60
        print(f"\n[{mins:6.1f} min] Último avance:\n{tail(log_path)}", flush=True)
        last_report = time.time()
    time.sleep(POLL_SECONDS)

print("\n[FIN] El pipeline terminó. Últimas líneas del log:", flush=True)
print(tail(log_path, n=15), flush=True)
print(f"\nResultados en: {DRIVE_DIR if drive_ok else REPO_DIR}", flush=True)
print("Recuerda liberar la VM desde tu Mac con: colab stop -s pfc", flush=True)
