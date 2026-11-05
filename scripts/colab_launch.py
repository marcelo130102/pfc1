"""
Lanzador dual para Opción 2 (Colab CLI desde Mac + Ejecución en VM de Colab).

Uso desde tu Mac:
    # 1. Crear sesión con GPU A100 y High-RAM y montar Drive
    colab new -s pfc --gpu A100 --high-mem
    colab drivemount -s pfc

    # 2. Verificar LLM, LightRAG, Graphiti y dataset antes de gastar GPU en todo el pipeline:
    python3 scripts/colab_smoke_test.py -s pfc

    # 3. Piloto para ajustar sin tocar los datos finales (archivos piloto_*):
    python3 scripts/colab_launch.py -s pfc --pilot

    # 4. Con el diseño congelado en un commit, corrida final (borra sus checkpoints):
    python3 scripts/colab_launch.py -s pfc --fresh

    # Si la VM se pierde a mitad de camino, reanudar desde los checkpoints de Drive:
    python3 scripts/colab_launch.py -s pfc
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/marcelo130102/pfc1.git"
REPO_DIR = "/content/pfc1"
DRIVE_DIR = "/content/drive/MyDrive/PFC_RAG_Resultados"
POLL_SECONDS = 30
NEO4J_VERSION = "5.26.21"
NEO4J_HOME = f"/content/neo4j-community-{NEO4J_VERSION}"
NEO4J_JAVA_HOME = "/usr/lib/jvm/java-17-openjdk-amd64"
# Neo4j solo escucha en localhost de la VM: esta contraseña no protege nada expuesto
NEO4J_PASSWORD = "pfc-local-neo4j"
REPORT_EVERY_SECONDS = 120

PACKAGES = [
    "lightrag-hku==1.5.7", "spacy", "vllm", "graphiti-core==0.31.2", "datasets", "pandas",
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


def ensure_neo4j_on_vm() -> None:
    """Instala y arranca Neo4j Community en la VM si aún no está escuchando."""

    def _bolt_is_open() -> bool:
        with socket.socket() as sock:
            sock.settimeout(2)
            return sock.connect_ex(("127.0.0.1", 7687)) == 0

    if _bolt_is_open():
        print("[OK] Neo4j ya está activo en la VM.", flush=True)
        return

    run(["apt-get", "install", "-y", "-q", "openjdk-17-jre-headless"])
    env = {**os.environ, "JAVA_HOME": NEO4J_JAVA_HOME}
    if not os.path.isdir(NEO4J_HOME):
        tarball = f"{NEO4J_HOME}.tar.gz"
        run(["curl", "-fsSL", "-o", tarball,
             f"https://dist.neo4j.org/neo4j-community-{NEO4J_VERSION}-unix.tar.gz"])
        run(["tar", "xzf", tarball, "-C", os.path.dirname(NEO4J_HOME)])
        run([f"{NEO4J_HOME}/bin/neo4j-admin", "dbms", "set-initial-password", NEO4J_PASSWORD], env=env)
        with open(f"{NEO4J_HOME}/conf/neo4j.conf", "a", encoding="utf-8") as conf:
            conf.write(
                "\ndbms.usage_report.enabled=false\n"
                "server.memory.heap.initial_size=2g\n"
                "server.memory.heap.max_size=8g\n"
                "server.memory.pagecache.size=4g\n"
            )
    run([f"{NEO4J_HOME}/bin/neo4j", "start"], env=env)

    for _ in range(60):
        if _bolt_is_open():
            print(f"[OK] Neo4j {NEO4J_VERSION} activo en la VM.", flush=True)
            return
        time.sleep(2)
    raise RuntimeError(f"Neo4j no arrancó. Revisa {NEO4J_HOME}/logs/neo4j.log")


def start_on_colab_vm() -> None:
    """
    Se ejecuta DENTRO de la VM de Colab:
    Instala dependencias, gestiona los checkpoints según --fresh y lanza
    `run_all_experiments.py` en segundo plano. El código ya lo subió el Mac.
    """
    drive_ok = os.path.isdir("/content/drive/MyDrive")
    if not drive_ok:
        print("[AVISO] Google Drive NO está montado. Ejecuta antes: colab drivemount -s pfc", flush=True)
    log_dir = DRIVE_DIR if drive_ok else REPO_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "experiments.log")

    # El piloto siempre empieza de cero, pero solo borra sus propios archivos
    pilot = os.environ.get("PFC_PILOT") == "1"
    mode_fresh = os.environ.get("PFC_FRESH") == "1" or pilot

    pid = running_pid()
    if pid and mode_fresh:
        print(f"[INFO] Deteniendo corrida anterior (PID {pid})...", flush=True)
        subprocess.run(["kill", "-9", str(pid)], check=False)
        time.sleep(2)
        pid = None

    if pid:
        print(f"[INFO] El pipeline ya estaba corriendo en la VM (PID {pid}).", flush=True)
    else:
        run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
        run([sys.executable, "-m", "spacy", "download", "en_core_web_lg", "-q"])
        ensure_neo4j_on_vm()

        env = os.environ.copy()
        env["PFC_NEO4J_VERSION"] = NEO4J_VERSION
        if mode_fresh:
            env["PFC_RESUME"] = "0"
            for old_output in (
                "experimento1_resultados.csv",
                "experimento1_calidad.csv",
                "experimento2_resultados.csv",
                "resumen_estadistico.csv",
                "muestra_revision_juez.csv",
                "run_manifest.json",
            ):
                for folder in (DRIVE_DIR, REPO_DIR):
                    p = os.path.join(folder, f"piloto_{old_output}" if pilot else old_output)
                    if os.path.exists(p):
                        os.remove(p)
                        print(f"[FRESH] Checkpoint previo eliminado: {p}", flush=True)
            if os.path.exists(log_path) and not pilot:
                os.remove(log_path)

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
    Sube los módulos locales (`src/*.py`, `scripts/*.py` y `run_all_experiments.py`) desde tu
    Mac hacia `/content/pfc1` en la VM para garantizar que corra siempre la última versión
    sin depender de `git push`.
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

    files_to_upload = [(repo_root / "run_all_experiments.py", f"{REPO_DIR}/run_all_experiments.py")]
    for folder in ("src", "scripts"):
        for local_f in sorted((repo_root / folder).glob("*.py")):
            files_to_upload.append((local_f, f"{REPO_DIR}/{folder}/{local_f.name}"))

    print("[SYNC] Subiendo código desde tu Mac a la VM de Colab...", flush=True)
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
        "--pilot",
        action="store_true",
        help="Corrida corta con otra semilla, preguntas reservadas y archivos piloto_*.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Elimina todos los checkpoints previos y recalcula los dos experimentos desde cero.",
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
            f"os.environ['PFC_PILOT'] = '{'1' if args.pilot else '0'}'\n"
        )
        full_code = header + script_path.read_text(encoding="utf-8")

        print(f"[COLAB] Inicializando pipeline en la sesión '{args.session}'...", flush=True)
        # El tiempo de espera cubre pip install: si venciera antes de lanzar el pipeline,
        # el monitoreo creería que ya terminó.
        started = subprocess.run(
            ["colab", "exec", "-s", args.session, "--timeout", "1800"],
            input=full_code,
            text=True,
            cwd=str(repo_root),
            check=False,
        )
        if started.returncode != 0:
            raise RuntimeError("No se pudo lanzar el pipeline en la VM.")

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
