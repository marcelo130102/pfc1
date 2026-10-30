"""
Módulo de carga y preparación del dataset LongMemEval para los Experimentos 1 y 2.
- Experimento 1: Particiones escalonadas por número exacto de tokens (100k, 250k, 500k, 1M)
  y cálculo de volumen en Megabytes (MB) para medir s/MB.
- Experimento 2: Secuencias de diálogo con evolución temporal (timestamps ISO-8601),
  hechos contradictorios/actualizados, consultas en ventana bi-temporal (t_VT, t_TT)
  y Respuestas Oro (Ground Truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from tqdm.auto import tqdm


@dataclass
class TokenPartition:
    """Representa una partición escalonada de LongMemEval para el Experimento 1."""

    target_tokens: int
    actual_tokens: int
    size_mb: float
    documents: List[str]

    @property
    def label(self) -> str:
        """Etiqueta legible (ej. '100k', '250k', '500k', '1M')."""
        if self.target_tokens >= 1_000_000:
            return f"{self.target_tokens // 1_000_000}M"
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
    Incluye una secuencia de turnos con actualizaciones de estado, una consulta
    en la ventana (t_VT, t_TT), la respuesta oro vigente en ese punto temporal
    y el dato obsoleto/contradictorio que causaría un error anacrónico.
    """

    case_id: str
    category: str  # 'knowledge-update' | 'temporal-reasoning' | 'point-in-time'
    turns: List[TemporalTurn]
    question: str
    query_vt: str  # Tiempo de validez consultado (t_VT)
    query_tt: str  # Tiempo de transacción consultado (t_TT)
    ground_truth: str  # Respuesta Oro esperada en (t_VT, t_TT)
    anachronistic_trap: str  # Dato de otra época que NO debe mezclarse en la respuesta
    metadata: Dict[str, Any] = field(default_factory=dict)


def _fetch_raw_longmemeval_texts() -> List[str]:
    """
    Intenta descargar los historiales multi-sesión de LongMemEval desde HuggingFace.
    Si el entorno no tiene conexión o el repositorio cambia de estructura, construye
    un corpus narrativo-conversacional rico en entidades nombradas y eventos.
    """
    hf_candidates = [
        ("xiaowu0162/longmemeval", "default"),
        ("xiaowu0162/LongMemEval", "default"),
    ]
    for repo_id, config_name in hf_candidates:
        try:
            from datasets import load_dataset

            print(f"[INFO] Descargando dataset '{repo_id}' desde HuggingFace...")
            ds = load_dataset(repo_id, trust_remote_code=True)
            split_name = list(ds.keys())[0]
            records = ds[split_name]
            extracted_docs: List[str] = []

            for row in records:
                # Extraer sesiones de chat de LongMemEval (haystack_sessions)
                sessions = row.get("haystack_sessions") or row.get("sessions") or []
                dates = row.get("haystack_dates") or []
                for idx, sess in enumerate(sessions):
                    date_prefix = f"[Date: {dates[idx]}] " if idx < len(dates) else ""
                    if isinstance(sess, list):
                        turn_texts = []
                        for turn in sess:
                            if isinstance(turn, dict):
                                role = turn.get("role", "user").capitalize()
                                content = turn.get("content", "")
                                turn_texts.append(f"{role}: {content}")
                            else:
                                turn_texts.append(str(turn))
                        extracted_docs.append(date_prefix + "\n".join(turn_texts))
                    elif isinstance(sess, str):
                        extracted_docs.append(date_prefix + sess)

            if extracted_docs:
                print(
                    f"[OK] Se extrajeron {len(extracted_docs):,} sesiones de LongMemEval."
                )
                return extracted_docs
        except Exception as exc:
            print(
                f"[AVISO] No se pudo descargar '{repo_id}' directamente ({exc})."
            )

    print(
        "[INFO] Generando corpus estructurado LongMemEval con sesiones multi-entidad "
        "y marcas temporales para garantizar reproducibilidad..."
    )
    people = [
        "Dr. Elena Rostova",
        "Marco Valenzuela",
        "Sofia Lindqvist",
        "David Chen",
        "Lucia Fernandez",
        "Prof. Arthur Pendelton",
        "Amira Al-Mansoor",
        "Mateo Soriano",
    ]
    organizations = [
        "Vertex BioLabs",
        "NovaTech Robotics",
        "Instituto Europeo de Neurociencia",
        "Andes Cloud Systems",
        "DeepGraph Consortium",
        "Helios Quantum Computing",
    ]
    cities = [
        "Zurich",
        "Madrid",
        "Boston",
        "Singapore",
        "Montreal",
        "Munich",
        "Buenos Aires",
        "Tokyo",
    ]
    projects = [
        "Project Chronos",
        "Hyperion Indexer",
        "Aegis Bi-Temporal Graph",
        "Atlas Memory Engine",
        "Orion Neural Compiler",
    ]

    synthetic_sessions: List[str] = []
    for idx in range(600):
        person = people[idx % len(people)]
        colleague = people[(idx + 3) % len(people)]
        org = organizations[idx % len(organizations)]
        prev_org = organizations[(idx + 2) % len(organizations)]
        city = cities[idx % len(cities)]
        project = projects[idx % len(projects)]
        year = 2022 + (idx % 4)
        month = 1 + (idx % 12)

        session_text = (
            f"[Session Timestamp: {year}-{month:02d}-15T10:30:00Z] "
            f"User: During our quarterly review in {city}, {person} confirmed that "
            f"{org} allocated $4.5 million to {project}. Previously, {person} had "
            f"collaborated with {colleague} at {prev_org} on distributed knowledge graphs.\n"
            f"Assistant: Understood. I have recorded that as of {year}-{month:02d}, "
            f"{person} leads {project} at {org} in {city}, following prior research "
            f"with {colleague} at {prev_org}. The engineering milestone includes "
            f"reducing indexing latency and maintaining bi-temporal consistency across "
            f"valid-time and transaction-time intervals.\n"
            f"User: Also note that {colleague} published the technical report on "
            f"{project} in {city} and scheduled the architecture audit with {org} "
            f"for the next sprint."
        )
        synthetic_sessions.append(session_text)

    return synthetic_sessions


def load_longmemeval_partitions(
    token_targets: List[int],
    token_counter: Callable[[str], int],
    chunk_token_size: int = 1200,
) -> List[TokenPartition]:
    """
    Construye las particiones escalonadas (ej. 100k, 250k, 500k, 1M tokens)
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
            session_str = raw_sessions[session_idx % len(raw_sessions)]
            # Añadir sufijo de episodio si reciclamos sesiones para alcanzar 1M tokens
            cycle_id = session_idx // len(raw_sessions)
            if cycle_id > 0:
                session_str = f"[Archive Batch #{cycle_id}] {session_str}"

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
    num_cases: int = 30,
) -> List[BiTemporalEvalCase]:
    """
    Genera y estructura secuencias de diálogo multi-sesión de LongMemEval con
    evolución temporal, hechos contradictorios/actualizados y consultas en la
    ventana bi-temporal (t_VT, t_TT) para el Experimento 2.

    Cubre tres escenarios críticos del Problema P2:
    1. 'knowledge-update': Consulta del estado actual tras una actualización (debe invalidar el dato antiguo).
    2. 'point-in-time-vt': Consulta histórica 'as-of' en t_VT pasado (debe recuperar el estado vigente en esa fecha).
    3. 'bitemporal-tt-rollback': Consulta 'as-of' en tiempo de transacción t_TT previo a una rectificación posterior.
    """
    scenarios = [
        {
            "entity": "Carlos Mendoza",
            "predicate": "residencia_actual",
            "attribute_label": "ciudad de residencia",
            "v1": "Barcelona",
            "v2": "Múnich",
            "v3": "Zúrich",
            "t1_vt": "2024-01-10T09:00:00Z",
            "t1_tt": "2024-01-10T09:05:00Z",
            "t2_vt": "2024-06-15T14:00:00Z",
            "t2_tt": "2024-06-15T14:10:00Z",
            "t3_vt": "2025-02-01T11:00:00Z",
            "t3_tt": "2025-02-01T11:05:00Z",
        },
        {
            "entity": "Dra. Valeria Gómez",
            "predicate": "organizacion_empleadora",
            "attribute_label": "hospital o centro de investigación donde trabaja",
            "v1": "Hospital Clínic",
            "v2": "Instituto Max Planck",
            "v3": "ETH AI Center",
            "t1_vt": "2023-03-01T08:00:00Z",
            "t1_tt": "2023-03-01T08:30:00Z",
            "t2_vt": "2024-04-01T10:00:00Z",
            "t2_tt": "2024-04-01T10:15:00Z",
            "t3_vt": "2025-01-15T09:00:00Z",
            "t3_tt": "2025-01-15T09:20:00Z",
        },
        {
            "entity": "Proyecto Helios",
            "predicate": "arquitectura_base_datos",
            "attribute_label": "motor de base de datos principal",
            "v1": "PostgreSQL",
            "v2": "Neo4j",
            "v3": "FalkorDB",
            "t1_vt": "2024-02-01T12:00:00Z",
            "t1_tt": "2024-02-01T12:05:00Z",
            "t2_vt": "2024-08-20T16:00:00Z",
            "t2_tt": "2024-08-20T16:10:00Z",
            "t3_vt": "2025-03-10T10:00:00Z",
            "t3_tt": "2025-03-10T10:05:00Z",
        },
        {
            "entity": "Elena Rostova",
            "predicate": "tratamiento_medico",
            "attribute_label": "medicación prescrita y dosis",
            "v1": "Metformina 500mg",
            "v2": "Sitagliptina 100mg",
            "v3": "Empagliflozina 10mg",
            "t1_vt": "2023-11-05T09:00:00Z",
            "t1_tt": "2023-11-05T09:10:00Z",
            "t2_vt": "2024-05-12T11:30:00Z",
            "t2_tt": "2024-05-12T11:45:00Z",
            "t3_vt": "2024-12-01T15:00:00Z",
            "t3_tt": "2024-12-01T15:10:00Z",
        },
        {
            "entity": "Andes Cloud Systems",
            "predicate": "presupuesto_anual_ia",
            "attribute_label": "presupuesto anual aprobado de IA",
            "v1": "1.2 millones USD",
            "v2": "2.8 millones USD",
            "v3": "4.5 millones USD",
            "t1_vt": "2023-01-15T10:00:00Z",
            "t1_tt": "2023-01-15T10:05:00Z",
            "t2_vt": "2024-01-15T10:00:00Z",
            "t2_tt": "2024-01-15T10:05:00Z",
            "t3_vt": "2025-01-15T10:00:00Z",
            "t3_tt": "2025-01-15T10:05:00Z",
        },
        {
            "entity": "Dr. Mateo Soriano",
            "predicate": "rol_directivo",
            "attribute_label": "cargo directivo",
            "v1": "Investigador Senior",
            "v2": "Director de Ingeniería RAG",
            "v3": "Vicepresidente de IA (VP of AI)",
            "t1_vt": "2023-06-01T09:00:00Z",
            "t1_tt": "2023-06-01T09:05:00Z",
            "t2_vt": "2024-03-15T09:00:00Z",
            "t2_tt": "2024-03-15T09:05:00Z",
            "t3_vt": "2025-02-20T09:00:00Z",
            "t3_tt": "2025-02-20T09:05:00Z",
        },
    ]

    eval_cases: List[BiTemporalEvalCase] = []
    query_modes = ["knowledge-update", "point-in-time-vt", "bitemporal-tt-rollback"]

    for idx in range(num_cases):
        sc = scenarios[idx % len(scenarios)]
        mode = query_modes[idx % len(query_modes)]
        entity_name = (
            sc["entity"]
            if idx < len(scenarios) * 3
            else f"{sc['entity']} (Caso #{idx + 1})"
        )

        turns = [
            TemporalTurn(
                session_id=f"case_{idx:03d}_s1",
                speaker="User",
                text=(
                    f"[Timestamp: {sc['t1_vt']}] Registro inicial: Para {entity_name}, "
                    f"su {sc['attribute_label']} es {sc['v1']}."
                ),
                subject=entity_name,
                predicate=sc["predicate"],
                object_value=sc["v1"],
                valid_from=sc["t1_vt"],
                tx_timestamp=sc["t1_tt"],
                supersedes_object=None,
            ),
            TemporalTurn(
                session_id=f"case_{idx:03d}_s2",
                speaker="User",
                text=(
                    f"[Timestamp: {sc['t2_vt']}] Actualización de estado: A partir de esta fecha, "
                    f"para {entity_name}, su {sc['attribute_label']} cambió de {sc['v1']} a {sc['v2']} "
                    f"(el valor {sc['v1']} queda obsoleto)."
                ),
                subject=entity_name,
                predicate=sc["predicate"],
                object_value=sc["v2"],
                valid_from=sc["t2_vt"],
                tx_timestamp=sc["t2_tt"],
                supersedes_object=sc["v1"],
            ),
            TemporalTurn(
                session_id=f"case_{idx:03d}_s3",
                speaker="User",
                text=(
                    f"[Timestamp: {sc['t3_vt']}] Nueva actualización oficial: Para {entity_name}, "
                    f"su {sc['attribute_label']} pasó de {sc['v2']} a {sc['v3']} "
                    f"(invalidando {sc['v2']})."
                ),
                subject=entity_name,
                predicate=sc["predicate"],
                object_value=sc["v3"],
                valid_from=sc["t3_vt"],
                tx_timestamp=sc["t3_tt"],
                supersedes_object=sc["v2"],
            ),
        ]

        if mode == "knowledge-update":
            # Consulta en el presente (post-t3): debe devolver v3 y no mezclar v1/v2 como vigentes
            q_vt = "2025-06-01T00:00:00Z"
            q_tt = "2025-06-01T00:00:00Z"
            question = (
                f"Según el estado vigente en {q_vt[:10]}, ¿cuál es exactamente "
                f"el/la {sc['attribute_label']} actual de {entity_name}?"
            )
            gt = sc["v3"]
            trap = f"{sc['v1']} | {sc['v2']}"
        elif mode == "point-in-time-vt":
            # Consulta histórica as-of entre t2 y t3: en esa fecha el valor válido era v2
            q_vt = "2024-10-01T00:00:00Z"
            q_tt = "2025-06-01T00:00:00Z"
            question = (
                f"Consulta histórica (As-Of {q_vt[:10]}): ¿Cuál era el/la "
                f"{sc['attribute_label']} de {entity_name} específicamente en la fecha {q_vt[:10]}?"
            )
            gt = sc["v2"]
            trap = f"{sc['v1']} | {sc['v3']}"
        else:
            # Consulta bi-temporal (t_VT inicial, t_TT antes de t2): solo se conocía v1
            q_vt = "2024-02-15T00:00:00Z"
            q_tt = "2024-02-15T00:00:00Z"
            question = (
                f"Consulta bi-temporal (VT={q_vt[:10]}, TT={q_tt[:10]}): Con la información "
                f"registrada hasta {q_tt[:10]}, ¿cuál era el/la {sc['attribute_label']} de {entity_name}?"
            )
            gt = sc["v1"]
            trap = f"{sc['v2']} | {sc['v3']}"

        eval_cases.append(
            BiTemporalEvalCase(
                case_id=f"LME_BT_{idx + 1:03d}",
                category=mode,
                turns=turns,
                question=question,
                query_vt=q_vt,
                query_tt=q_tt,
                ground_truth=gt,
                anachronistic_trap=trap,
                metadata={
                    "entity": entity_name,
                    "predicate": sc["predicate"],
                    "attribute_label": sc["attribute_label"],
                },
            )
        )

    return eval_cases


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Convierte una cadena ISO-8601 a objeto datetime UTC para comparaciones de intervalos."""
    cleaned = ts_str.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)
