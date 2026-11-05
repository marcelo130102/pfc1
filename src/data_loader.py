"""
Módulo de carga y preparación del dataset LongMemEval para los Experimentos 1 y 2.
- Experimento 1: Particiones escalonadas por número exacto de tokens (100k, 250k, 500k,
  1M, 1.5M, 2.5M) y cálculo de volumen en Megabytes (MB) para medir s/MB.
- Experimento 2: 100 secuencias de diálogo con evolución temporal profunda (5 estados
  t1 -> t2 -> t3 -> t4 -> t5), hechos contradictorios/actualizados, 4 categorías de
  consultas en ventana bi-temporal (t_VT, t_TT) y Respuestas Oro (Ground Truth).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from tqdm.auto import tqdm

# Versión depurada que, según su README, reemplaza al LongMemEval original.
LONGMEMEVAL_REPO_ID = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_FILENAME = "longmemeval_oracle.json"

# Se completa al cargar el corpus; run_all_experiments.py lo guarda en el manifiesto.
DATASET_PROVENANCE: Dict[str, Any] = {}


@dataclass
class TokenPartition:
    """Representa una partición escalonada de LongMemEval para el Experimento 1."""

    target_tokens: int
    actual_tokens: int
    size_mb: float
    documents: List[str]

    @property
    def label(self) -> str:
        """Etiqueta legible (ej. '100k', '250k', '500k', '1M', '1.5M', '2.5M')."""
        if self.target_tokens >= 1_000_000:
            val_m = self.target_tokens / 1_000_000
            return f"{int(val_m)}M" if val_m.is_integer() else f"{val_m:.1f}M"
        return f"{self.target_tokens // 1_000}k"


@dataclass
class TemporalTurn:
    """Evento o sesión de diálogo individual con tiempos de validez (VT) y transacción (TT)."""

    session_id: str
    speaker: str
    text: str
    subject: str
    predicate: str
    object_value: str
    valid_from: str  # ISO-8601 (Inicio de validez del hecho en el mundo real: t_VT)
    tx_timestamp: str  # ISO-8601 (Estampa de tiempo de ingesta en el sistema: t_TT)
    supersedes_object: Optional[str] = None


@dataclass
class BiTemporalEvalCase:
    """
    Caso de evaluación para el Experimento 2 (Recuperación Bi-temporal).
    Incluye una secuencia de 5 turnos con actualizaciones sucesivas de estado, una consulta
    en la ventana (t_VT, t_TT), la respuesta oro vigente en ese punto temporal
    y los datos obsoletos/futuros que causarían un error anacrónico.
    """

    case_id: str
    category: str  # 'knowledge-update' | 'point-in-time-vt' | 'bitemporal-tt-rollback' | 'retroactive-audit'
    turns: List[TemporalTurn]
    question: str
    query_vt: str  # Tiempo de validez consultado (t_VT)
    query_tt: str  # Tiempo de transacción consultado (t_TT)
    ground_truth: str  # Respuesta Oro esperada en (t_VT, t_TT)
    anachronistic_trap: str  # Datos de otras épocas que NO deben mezclarse en la respuesta
    metadata: Dict[str, Any] = field(default_factory=dict)


def _fetch_raw_longmemeval_texts() -> List[str]:
    """
    Descarga los historiales multi-sesión de LongMemEval desde HuggingFace Hub y
    devuelve una sesión única por documento. Cualquier fallo de descarga o de formato
    detiene la corrida: los resultados nunca se calculan sobre otro corpus.
    """
    from huggingface_hub import hf_hub_download

    local_json = hf_hub_download(
        repo_id=LONGMEMEVAL_REPO_ID,
        filename=LONGMEMEVAL_FILENAME,
        repo_type="dataset",
    )
    with open(local_json, "rb") as fh:
        raw_bytes = fh.read()
    records = json.loads(raw_bytes)

    # Una misma sesión puede pertenecer a varias preguntas: se conserva una sola copia
    sessions_by_id: Dict[str, str] = {}
    for row in records:
        for session_id, date, session in zip(
            row["haystack_session_ids"],
            row["haystack_dates"],
            row["haystack_sessions"],
        ):
            if session_id in sessions_by_id:
                continue
            turn_texts = [
                f"{turn['role'].capitalize()}: {turn['content']}" for turn in session
            ]
            sessions_by_id[session_id] = f"[Date: {date}] " + "\n".join(turn_texts)

    if not sessions_by_id:
        raise ValueError(
            f"{LONGMEMEVAL_REPO_ID}/{LONGMEMEVAL_FILENAME} no contiene sesiones."
        )

    DATASET_PROVENANCE.update(
        repo_id=LONGMEMEVAL_REPO_ID,
        filename=LONGMEMEVAL_FILENAME,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        unique_sessions=len(sessions_by_id),
    )
    print(
        f"[OK] Se extrajeron {len(sessions_by_id):,} sesiones reales de LongMemEval "
        f"({LONGMEMEVAL_REPO_ID}/{LONGMEMEVAL_FILENAME})."
    )
    return list(sessions_by_id.values())


def load_longmemeval_partitions(
    token_targets: List[int],
    token_counter: Callable[[str], int],
    chunk_token_size: int = 1200,
) -> List[TokenPartition]:
    """
    Construye las particiones escalonadas (100k, 250k, 500k, 1M, 1.5M, 2.5M tokens)
    para el Experimento 1, agrupadas en documentos de ~chunk_token_size tokens.
    Calcula el peso exacto en Megabytes (MB) de cada partición.
    """
    raw_sessions = _fetch_raw_longmemeval_texts()
    partitions: List[TokenPartition] = []

    for target in tqdm(
        token_targets, desc="Preparando particiones escalonadas LongMemEval"
    ):
        docs: List[str] = []
        current_chunk_lines: List[str] = []
        current_chunk_tokens = 0
        accumulated_tokens = 0
        session_idx = 0

        while accumulated_tokens < target:
            # Repetir sesiones inflaría el tamaño con contenido duplicado
            if session_idx >= len(raw_sessions):
                raise ValueError(
                    f"El corpus solo tiene {accumulated_tokens:,} tokens, insuficientes "
                    f"para la partición de {target:,} tokens."
                )
            session_str = raw_sessions[session_idx]

            tok_len = token_counter(session_str)
            current_chunk_lines.append(session_str)
            current_chunk_tokens += tok_len
            accumulated_tokens += tok_len
            session_idx += 1

            if current_chunk_tokens >= chunk_token_size or accumulated_tokens >= target:
                docs.append("\n\n".join(current_chunk_lines))
                current_chunk_lines = []
                current_chunk_tokens = 0

        total_bytes = sum(len(d.encode("utf-8")) for d in docs)
        size_mb = max(1e-6, total_bytes / (1024 * 1024))

        partitions.append(
            TokenPartition(
                target_tokens=target,
                actual_tokens=accumulated_tokens,
                size_mb=round(size_mb, 4),
                documents=docs,
            )
        )

    return partitions


def load_bitemporal_evaluation_cases(
    num_cases: int = 100,
) -> List[BiTemporalEvalCase]:
    """
    Genera 100 secuencias de diálogo sintéticas (escenarios definidos aquí, no
    provenientes de LongMemEval) con evolución temporal profunda de 5 estados (t1 -> t2 -> t3 -> t4 -> t5), hechos
    contradictorios/actualizados y consultas en la ventana bi-temporal (t_VT, t_TT)
    para el Experimento 2.

    Cubre cuatro escenarios críticos del Problema P2:
    1. 'knowledge-update': Estado actual tras 5 actualizaciones (debe devolver v5 e invalidar v1..v4).
    2. 'point-in-time-vt': Consulta histórica 'as-of' en t_VT intermedio [t3, t4) (debe recuperar v3).
    3. 'bitemporal-tt-rollback': Consulta en tiempo de transacción t_TT entre [t2, t3) (debe recuperar v2).
    4. 'retroactive-audit': Auditoría del estado fundacional en la ventana inicial [t1, t2) (debe recuperar v1).
    """
    scenarios = [
        {
            "entity": "Carlos Mendoza",
            "predicate": "residencia_actual",
            "attribute_label": "ciudad de residencia",
            "values": ["Barcelona", "Múnich", "Zúrich", "Estocolmo", "Singapur"],
        },
        {
            "entity": "Dra. Valeria Gómez",
            "predicate": "organizacion_empleadora",
            "attribute_label": "centro de investigación donde trabaja",
            "values": [
                "Hospital Clínic",
                "Instituto Max Planck",
                "ETH AI Center",
                "MIT CSAIL",
                "DeepMind Zurich",
            ],
        },
        {
            "entity": "Proyecto Helios",
            "predicate": "arquitectura_base_datos",
            "attribute_label": "motor de base de datos principal",
            "values": ["PostgreSQL", "MongoDB", "Neo4j", "FalkorDB", "Spanner Graph"],
        },
        {
            "entity": "Elena Rostova",
            "predicate": "tratamiento_medico",
            "attribute_label": "medicación prescrita y dosis",
            "values": [
                "Metformina 500mg",
                "Metformina 850mg",
                "Sitagliptina 100mg",
                "Empagliflozina 10mg",
                "Semaglutida 1mg",
            ],
        },
        {
            "entity": "Andes Cloud Systems",
            "predicate": "presupuesto_anual_ia",
            "attribute_label": "presupuesto anual aprobado de IA",
            "values": [
                "1.2 millones USD",
                "2.0 millones USD",
                "2.8 millones USD",
                "4.5 millones USD",
                "7.0 millones USD",
            ],
        },
        {
            "entity": "Dr. Mateo Soriano",
            "predicate": "rol_directivo",
            "attribute_label": "cargo directivo",
            "values": [
                "Investigador Senior",
                "Líder Técnico de Grafos",
                "Director de Ingeniería RAG",
                "Vicepresidente de IA",
                "Chief AI Officer (CAIO)",
            ],
        },
        {
            "entity": "Plataforma Chronos",
            "predicate": "cluster_entrenamiento",
            "attribute_label": "acelerador de hardware asignado",
            "values": [
                "NVIDIA T4 16GB",
                "NVIDIA A10G 24GB",
                "NVIDIA L4 24GB",
                "NVIDIA A100 80GB",
                "NVIDIA H100 80GB",
            ],
        },
        {
            "entity": "Consorcio Aegis",
            "predicate": "protocolo_cifrado",
            "attribute_label": "estándar criptográfico activo",
            "values": [
                "RSA-2048",
                "RSA-4096",
                "ECDSA P-256",
                "Ed25519",
                "ML-KEM-768 Post-Quantum",
            ],
        },
        {
            "entity": "NovaTech Robotics",
            "predicate": "sede_corporativa",
            "attribute_label": "sede fiscal corporativa",
            "values": ["Madrid", "Dublín", "Ámsterdam", "Londres", "Boston"],
        },
        {
            "entity": "Motor Atlas RAG",
            "predicate": "modelo_generativo_base",
            "attribute_label": "modelo LLM de producción",
            "values": [
                "Llama-2-7B",
                "Mistral-7B-v0.2",
                "Llama-3-8B",
                "Qwen2.5-7B-Instruct",
                "Llama-3.1-70B-Instruct",
            ],
        },
    ]

    # 5 marcas temporales escalonadas (VT y TT) para las 5 transiciones de estado
    timestamps = [
        ("2023-01-15T09:00:00Z", "2023-01-15T09:05:00Z"),  # t1 (v1)
        ("2023-07-20T11:00:00Z", "2023-07-20T11:05:00Z"),  # t2 (v2)
        ("2024-02-10T14:00:00Z", "2024-02-10T14:05:00Z"),  # t3 (v3)
        ("2024-09-05T16:00:00Z", "2024-09-05T16:05:00Z"),  # t4 (v4)
        ("2025-03-01T10:00:00Z", "2025-03-01T10:05:00Z"),  # t5 (v5)
    ]

    eval_cases: List[BiTemporalEvalCase] = []
    query_modes = [
        "knowledge-update",
        "point-in-time-vt",
        "bitemporal-tt-rollback",
        "retroactive-audit",
    ]

    for idx in range(num_cases):
        sc = scenarios[idx % len(scenarios)]
        mode = query_modes[idx % len(query_modes)]
        vals = sc["values"]
        entity_name = (
            sc["entity"]
            if idx < len(scenarios)
            else f"{sc['entity']} (Exp #{idx + 1})"
        )

        turns: List[TemporalTurn] = []
        for step_i, val in enumerate(vals):
            t_vt, t_tt = timestamps[step_i]
            prev_val = vals[step_i - 1] if step_i > 0 else None
            if step_i == 0:
                turn_txt = (
                    f"[Timestamp: {t_vt}] Registro inicial: Para {entity_name}, "
                    f"su {sc['attribute_label']} es {val}."
                )
            else:
                turn_txt = (
                    f"[Timestamp: {t_vt}] Actualización #{step_i}: A partir de esta fecha, "
                    f"para {entity_name}, su {sc['attribute_label']} cambió de {prev_val} a {val} "
                    f"(quedando {prev_val} obsoleto)."
                )
            turns.append(
                TemporalTurn(
                    session_id=f"case_{idx:03d}_s{step_i + 1}",
                    speaker="User",
                    text=turn_txt,
                    subject=entity_name,
                    predicate=sc["predicate"],
                    object_value=val,
                    valid_from=t_vt,
                    tx_timestamp=t_tt,
                    supersedes_object=prev_val,
                )
            )

        if mode == "knowledge-update":
            # Consulta tras las 5 actualizaciones: respuesta vigente v5, trampas v1..v4
            q_vt = "2025-06-15T00:00:00Z"
            q_tt = "2025-06-15T00:00:00Z"
            question = (
                f"Según el estado vigente en {q_vt[:10]}, ¿cuál es exactamente "
                f"el/la {sc['attribute_label']} actual de {entity_name}?"
            )
            gt = vals[4]
            traps = [vals[0], vals[1], vals[2], vals[3]]
        elif mode == "point-in-time-vt":
            # Consulta histórica as-of entre t3 y t4 (ej. mayo 2024): vigente era v3
            q_vt = "2024-05-20T00:00:00Z"
            q_tt = "2025-06-15T00:00:00Z"
            question = (
                f"Consulta histórica (As-Of {q_vt[:10]}): ¿Cuál era el/la "
                f"{sc['attribute_label']} de {entity_name} específicamente en la fecha {q_vt[:10]}?"
            )
            gt = vals[2]
            traps = [vals[0], vals[1], vals[3], vals[4]]
        elif mode == "bitemporal-tt-rollback":
            # Consulta bi-temporal con corte de transacción entre t2 y t3 (noviembre 2023): v2
            q_vt = "2023-11-10T00:00:00Z"
            q_tt = "2023-11-10T00:00:00Z"
            question = (
                f"Consulta bi-temporal (VT={q_vt[:10]}, TT={q_tt[:10]}): Con la información "
                f"registrada en el sistema hasta {q_tt[:10]}, ¿cuál era el/la {sc['attribute_label']} de {entity_name}?"
            )
            gt = vals[1]
            traps = [vals[0], vals[2], vals[3], vals[4]]
        else:
            # 'retroactive-audit': Auditoría en la ventana inicial [t1, t2) (abril 2023): v1
            q_vt = "2023-04-01T00:00:00Z"
            q_tt = "2025-06-15T00:00:00Z"
            question = (
                f"Auditoría temporal (VT={q_vt[:10]}, TT={q_tt[:10]}): ¿Cuál era el valor "
                f"original vigente en {q_vt[:10]} para el/la {sc['attribute_label']} de {entity_name}?"
            )
            gt = vals[0]
            traps = [vals[1], vals[2], vals[3], vals[4]]

        eval_cases.append(
            BiTemporalEvalCase(
                case_id=f"LME_BT_{idx + 1:03d}",
                category=mode,
                turns=turns,
                question=question,
                query_vt=q_vt,
                query_tt=q_tt,
                ground_truth=gt,
                anachronistic_trap=" | ".join(traps),
                metadata={
                    "entity": entity_name,
                    "predicate": sc["predicate"],
                    "attribute_label": sc["attribute_label"],
                    "num_state_transitions": len(vals),
                },
            )
        )

    return eval_cases


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Convierte una cadena ISO-8601 a objeto datetime UTC para comparaciones de intervalos."""
    cleaned = ts_str.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)
