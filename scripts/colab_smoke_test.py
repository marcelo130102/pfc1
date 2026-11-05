"""
Prueba corta de la VM de Colab antes de gastar GPU en el pipeline completo.
Verifica con llamadas reales: dataset, servidor vLLM, LLM, embeddings, Juez LLM,
aislamiento y consulta de LightRAG, E2GraphRAG y un escenario bi-temporal en Graphiti. Termina con código 1 si alguna verificación falla.

Uso desde tu Mac (la sesión debe existir):
    colab new -s pfc --gpu A100 --high-mem
    python3 scripts/colab_smoke_test.py -s pfc

Al terminar descarga smoke_test.log y vllm_server.log, y deja la VM encendida para
poder lanzar el pipeline. Libérala con: colab stop -s pfc
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_launch import (
    PACKAGES,
    REPO_DIR,
    colab_exec_short,
    ensure_neo4j_on_vm,
    sync_local_code_to_vm,
)

LOG_NAME = "smoke_test.log"
PASS_MARKER = "SMOKE:PASS"
FAIL_MARKER = "SMOKE:FAIL"
POLL_SECONDS = 20
REPORT_EVERY_SECONDS = 60
MAX_WAIT_SECONDS = 45 * 60

# Texto propio con entidades explícitas: un LLM real siempre debe extraer algo de aquí.
LIGHTRAG_SAMPLE_DOC = (
    "Marie Curie worked at the University of Paris, where she collaborated with "
    "Pierre Curie on radioactivity. In 1903 Marie Curie and Pierre Curie received "
    "the Nobel Prize in Physics in Stockholm together with Henri Becquerel."
)

START_SNIPPET = f"""
import os, subprocess, sys
log_p = "{REPO_DIR}/{LOG_NAME}"
if os.path.exists(log_p):
    os.remove(log_p)
log_fh = open(log_p, "a", encoding="utf-8")
subprocess.Popen(
    [sys.executable, "-u", "scripts/colab_smoke_test.py", "--vm"],
    cwd="{REPO_DIR}", stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True,
)
print("STARTED")
"""

POLL_SNIPPET = f"""
import os, subprocess
out = subprocess.run(["pgrep", "-f", "colab_smoke_test.py --vm"], capture_output=True, text=True).stdout.split()
log_p = "{REPO_DIR}/{LOG_NAME}"
lines = ""
if os.path.exists(log_p):
    with open(log_p, "r", encoding="utf-8", errors="replace") as f:
        lines = "".join(f.readlines()[-60:]).rstrip()
print("STATUS:RUNNING" if out else "STATUS:DONE")
print(lines)
"""


# =============================================================================
# Verificaciones (se ejecutan dentro de la VM)
# =============================================================================


async def check_dataset(config, engine):
    from src.data_loader import DATASET_PROVENANCE, load_longmemeval_partitions

    if engine._get_tokenizer() is None:
        raise RuntimeError("tiktoken no está disponible: el conteo de tokens sería aproximado.")
    largest = max(config.exp1_token_partitions)
    partition = load_longmemeval_partitions([largest], engine.count_tokens)[0]
    return (
        f"{DATASET_PROVENANCE} | partición {partition.label}: "
        f"{partition.actual_tokens:,} tokens, {len(partition.documents)} documentos"
    )


async def check_server(config, engine):
    if not engine.is_server_healthy():
        engine.start_vllm_server(timeout_seconds=600)
    with urllib.request.urlopen(f"{config.base_url}/models", timeout=10) as resp:
        models = [m["id"] for m in json.loads(resp.read().decode("utf-8"))["data"]]
    if config.llm_model_id not in models:
        raise RuntimeError(f"Modelo configurado {config.llm_model_id} no está en {models}")
    return f"modelos servidos: {models}"


async def check_llm(config, engine):
    engine.tracker.reset()
    answer = await engine.generate_async(
        "What is the capital of France? Answer in one word.", max_tokens=16
    )
    usage = engine.tracker.snapshot()
    if not answer or usage["completion_tokens"] == 0:
        raise RuntimeError(f"Respuesta vacía del LLM (uso: {usage})")
    return f"respuesta={answer!r} | uso={usage}"


async def check_embeddings(config, engine):
    import numpy as np

    vectors = engine.encode_texts(["Marie Curie won a Nobel Prize.", "The train leaves at noon."])
    if vectors.shape != (2, config.embedding_dim):
        raise RuntimeError(f"Dimensión inesperada de embeddings: {vectors.shape}")
    return f"shape={vectors.shape} | coseno={float(np.dot(vectors[0], vectors[1])):.3f}"


async def check_judge(config, engine):
    from src.evaluation import AnswerJudge

    judge = AnswerJudge(engine)
    question = "What is the employer of Ana Torres?"
    traps = ["Hospital Clinic", "ETH AI Center"]
    correct = await judge.evaluate(question, "Max Planck Institute", "Ana works at the Max Planck Institute.", traps)
    wrong = await judge.evaluate(question, "Max Planck Institute", "Ana works at Hospital Clinic.", traps)
    if correct["correct"] is not True or correct["anachronistic"] is not False:
        raise RuntimeError(f"El juez rechazó la respuesta correcta: {correct}")
    if wrong["correct"] is not False or wrong["anachronistic"] is not True:
        raise RuntimeError(f"El juez aceptó la respuesta anacrónica: {wrong}")
    return f"correcta={correct} | anacrónica={wrong}"


def _sample_partition(engine, target_tokens):
    from src.data_loader import TokenPartition

    return TokenPartition(
        target_tokens=target_tokens,
        actual_tokens=engine.count_tokens(LIGHTRAG_SAMPLE_DOC),
        size_mb=len(LIGHTRAG_SAMPLE_DOC.encode("utf-8")) / (1024 * 1024),
        documents=[LIGHTRAG_SAMPLE_DOC],
    )


async def check_lightrag(config, engine):
    from src.exp1_indexing import LightRAGBaselineIndexer

    indexer = LightRAGBaselineIndexer(config, engine)
    # El indexador lanza una excepción si el LLM falla o si el grafo queda vacío.
    # Dos particiones con el mismo documento: sin aislamiento, la segunda lo omitiría.
    first = await indexer.index_partition_async(_sample_partition(engine, 1_000), run_id=1)
    second = await indexer.index_partition_async(_sample_partition(engine, 2_000), run_id=1)
    if second["llm_calls"] == 0:
        raise RuntimeError("La segunda partición no llamó al LLM: el aislamiento falló.")
    answer = await indexer.answer("Who received the Nobel Prize with Pierre Curie?")
    await indexer.close()
    return (
        f"nodos={first['nodos_grafo']}/{second['nodos_grafo']} "
        f"llamadas_llm={first['llm_calls']}/{second['llm_calls']} | respuesta={answer[:120]!r}"
    )


async def check_e2graphrag(config, engine):
    from src.e2graphrag import E2GraphRAGIndexer

    indexer = E2GraphRAGIndexer(config, engine)
    metrics = await indexer.index_partition_async(_sample_partition(engine, 1_000))
    if metrics["nodos_grafo"] == 0:
        raise RuntimeError("spaCy no extrajo entidades del documento de prueba.")
    question = "Who received the Nobel Prize with Pierre Curie?"
    _, path = indexer.retrieve(question)
    answer = await indexer.answer(question)
    return (
        f"nodos={metrics['nodos_grafo']} aristas={metrics['aristas_grafo']} "
        f"camino={path} | respuesta={answer[:120]!r}"
    )


async def check_graphiti(config, engine):
    from src.data_loader import load_bitemporal_evaluation_cases
    from src.exp2_bitemporal import GraphitiMemory

    cases = load_bitemporal_evaluation_cases(num_scenarios=1)
    memory = GraphitiMemory(config, engine)
    lines = []
    try:
        await memory.open(cases[0].group_id)
        await memory.ingest(cases[0].episodes)
        edges = await memory.graphiti.search(
            cases[0].question, group_ids=[memory.group_id], num_results=config.exp2_candidate_pool
        )
        if not edges:
            raise RuntimeError("Graphiti no extrajo ningún hecho del escenario.")
        for edge in edges:
            lines.append(f"{edge.fact} | valid={edge.valid_at} invalid={edge.invalid_at} expired={edge.expired_at}")
        for case in cases:
            answer, facts = await memory.answer(case, as_of=True)
            lines.append(f"{case.category}: oro={case.ground_truth!r} hechos={len(facts)} respuesta={answer[:120]!r}")
    finally:
        await memory.close()
    # Se muestra sin exigir acierto: lo que responda Graphiti es justamente lo que se mide
    return "\n  " + "\n  ".join(lines)


async def run_checks() -> int:
    from src.config import ExperimentConfig
    from src.inference_setup import LocalInferenceEngine

    config = ExperimentConfig()
    print(f"[HARDWARE] {config.auto_configure_for_colab_gpu()}", flush=True)
    engine = LocalInferenceEngine(config)

    # Sin servidor no tiene sentido probar lo que depende de él
    checks = [
        ("Dataset LongMemEval", check_dataset, False),
        ("Servidor vLLM", check_server, False),
        ("Llamada real al LLM", check_llm, True),
        ("Embeddings bge-m3", check_embeddings, False),
        ("Juez LLM", check_judge, True),
        ("LightRAG: aislamiento y consulta", check_lightrag, True),
        ("E2GraphRAG: índice y consulta", check_e2graphrag, True),
        ("Graphiti: escenario bi-temporal", check_graphiti, True),
    ]

    results = {}
    server_ok = True
    for name, check, needs_server in checks:
        if needs_server and not server_ok:
            results[name] = False
            print(f"[SKIP] {name}: el servidor vLLM no está disponible", flush=True)
            continue
        print(f"\n--- {name} ---", flush=True)
        try:
            print(f"[PASS] {name}: {await check(config, engine)}", flush=True)
            results[name] = True
        except Exception:
            traceback.print_exc(file=sys.stdout)
            print(f"[FAIL] {name}", flush=True)
            results[name] = False
            if check is check_server:
                server_ok = False

    print("\n=== RESUMEN ===", flush=True)
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}", flush=True)
    all_passed = all(results.values())
    print(PASS_MARKER if all_passed else FAIL_MARKER, flush=True)
    return 0 if all_passed else 1


def run_checks_on_vm() -> int:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_lg", "-q"], check=True)
    ensure_neo4j_on_vm()
    # Un solo event loop, igual que run_all_experiments.py: el cliente async del motor
    # y los workers de LightRAG quedan ligados al loop en el que se crean.
    return asyncio.run(run_checks())


# =============================================================================
# Lanzamiento y monitoreo (se ejecutan en tu Mac)
# =============================================================================


def run_from_mac(session: str) -> int:
    sync_local_code_to_vm(session=session, repo_root=REPO_ROOT)

    started = colab_exec_short(session, START_SNIPPET, REPO_ROOT)
    if "STARTED" not in started.stdout:
        print(f"[ERROR] No se pudo iniciar la prueba en la VM:\n{started.stdout}\n{started.stderr}")
        return 1
    print("[OK] Prueba iniciada en la VM. Puede tardar varios minutos.", flush=True)

    t0 = time.time()
    last_report = 0.0
    output = ""
    while time.time() - t0 < MAX_WAIT_SECONDS:
        time.sleep(POLL_SECONDS)
        res = colab_exec_short(session, POLL_SNIPPET, REPO_ROOT)
        if res.returncode == 0:
            output = (res.stdout or "").strip()
        if "STATUS:DONE" in output:
            break
        if time.time() - last_report >= REPORT_EVERY_SECONDS:
            mins = (time.time() - t0) / 60.0
            tail_text = "\n".join(output.splitlines()[-4:])
            print(f"\n[{mins:5.1f} min] Progreso en VM:\n{tail_text}", flush=True)
            last_report = time.time()
    else:
        print(f"[ERROR] La prueba no terminó en {MAX_WAIT_SECONDS // 60} minutos.")

    print(f"\n{output.split('STATUS:DONE', 1)[-1].strip()}", flush=True)

    for filename in (LOG_NAME, "vllm_server.log"):
        res = subprocess.run(
            ["colab", "download", "-s", session, f"{REPO_DIR}/{filename}", str(REPO_ROOT / filename)],
            capture_output=True,
            text=True,
        )
        status = "descargado" if res.returncode == 0 else f"no disponible ({res.stderr.strip()})"
        print(f"[LOG] {filename}: {status}", flush=True)

    print(f"Libera la VM cuando termines con: colab stop -s {session}", flush=True)

    # Un log cortado no cuenta como éxito: se exige el marcador final
    passed = PASS_MARKER in output and "STATUS:DONE" in output
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica la VM de Colab antes del pipeline completo.")
    parser.add_argument("--session", "-s", default="pfc", help="Nombre de la sesión de Colab (default: pfc)")
    parser.add_argument("--vm", action="store_true", help="Uso interno: ejecuta las verificaciones en la VM")
    args = parser.parse_args()
    sys.exit(run_checks_on_vm() if args.vm else run_from_mac(args.session))
