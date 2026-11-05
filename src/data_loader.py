"""
Módulo de carga y preparación de datos para los Experimentos 1 y 2.
- Experimento 1: particiones escalonadas de sesiones reales de LongMemEval por número de
  tokens (100k a 2.5M), y preguntas de LongMemEval para medir la calidad de respuesta.
- Experimento 2: preguntas 'temporal-reasoning' y 'knowledge-update' de LongMemEval, más
  escenarios bi-temporales SINTÉTICOS con hechos tardíos y correcciones retroactivas
  (LongMemEval no distingue tiempo de validez y tiempo de transacción).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from tqdm.auto import tqdm

# Versión depurada que, según su README, reemplaza al LongMemEval original.
LONGMEMEVAL_REPO_ID = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_FILENAME = "longmemeval_oracle.json"

# Se completa al cargar el corpus; run_all_experiments.py lo guarda en el manifiesto.
DATASET_PROVENANCE: Dict[str, Any] = {}

_RECORDS_CACHE: List[Dict[str, Any]] = []


@dataclass
class TokenPartition:
    """Representa una partición escalonada de LongMemEval para el Experimento 1."""

    target_tokens: int
    actual_tokens: int
    size_mb: float
    documents: List[str]
    session_ids: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Etiqueta legible (ej. '100k', '250k', '500k', '1M', '1.5M', '2.5M')."""
        if self.target_tokens >= 1_000_000:
            val_m = self.target_tokens / 1_000_000
            return f"{int(val_m)}M" if val_m.is_integer() else f"{val_m:.1f}M"
        return f"{self.target_tokens // 1_000}k"


@dataclass
class TemporalEpisode:
    """Sesión ingerida en la fecha `recorded_at`, que es su tiempo de transacción (TT)."""

    episode_id: str
    text: str
    recorded_at: datetime


@dataclass
class EvalCase:
    """
    Pregunta del Experimento 2. Los casos con el mismo `group_id` comparten los mismos
    episodios y se responden sobre un único índice.
    En LongMemEval, `query_vt` y `query_tt` son la fecha de la pregunta.
    """

    case_id: str
    source: str  # 'longmemeval' | 'sintetico'
    group_id: str
    category: str
    episodes: List[TemporalEpisode]
    question: str
    ground_truth: str
    query_vt: Optional[datetime] = None
    query_tt: Optional[datetime] = None
    anachronistic_traps: List[str] = field(default_factory=list)


@dataclass
class QualityQuestion:
    """Pregunta de LongMemEval usada para medir la calidad de respuesta del Experimento 1."""

    question_id: str
    question_type: str
    question: str
    answer: str
    session_ids: List[str]


def is_pilot_question(question_id: str) -> bool:
    """Reserva de forma determinista cerca del 10% de las preguntas para el piloto."""
    return int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16) % 10 == 0


def _load_longmemeval_records() -> List[Dict[str, Any]]:
    """
    Descarga LongMemEval desde HuggingFace Hub. Cualquier fallo de descarga o de formato
    detiene la corrida: los resultados nunca se calculan sobre otro corpus.
    """
    if _RECORDS_CACHE:
        return _RECORDS_CACHE

    from huggingface_hub import hf_hub_download

    local_json = hf_hub_download(
        repo_id=LONGMEMEVAL_REPO_ID,
        filename=LONGMEMEVAL_FILENAME,
        repo_type="dataset",
    )
    with open(local_json, "rb") as fh:
        raw_bytes = fh.read()
    _RECORDS_CACHE.extend(json.loads(raw_bytes))
    DATASET_PROVENANCE.update(
        repo_id=LONGMEMEVAL_REPO_ID,
        filename=LONGMEMEVAL_FILENAME,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        questions=len(_RECORDS_CACHE),
    )
    return _RECORDS_CACHE


def _format_session(date: str, session: List[Dict[str, Any]]) -> str:
    turn_texts = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in session]
    return f"[Date: {date}] " + "\n".join(turn_texts)


def _fetch_raw_longmemeval_texts() -> Dict[str, str]:
    """Devuelve el texto de cada sesión única de LongMemEval, indexado por su identificador."""
    # Una misma sesión puede pertenecer a varias preguntas: se conserva una sola copia
    sessions_by_id: Dict[str, str] = {}
    for row in _load_longmemeval_records():
        for session_id, date, session in zip(
            row["haystack_session_ids"],
            row["haystack_dates"],
            row["haystack_sessions"],
        ):
            if session_id not in sessions_by_id:
                sessions_by_id[session_id] = _format_session(date, session)

    if not sessions_by_id:
        raise ValueError(
            f"{LONGMEMEVAL_REPO_ID}/{LONGMEMEVAL_FILENAME} no contiene sesiones."
        )
    DATASET_PROVENANCE["unique_sessions"] = len(sessions_by_id)
    print(
        f"[OK] Se extrajeron {len(sessions_by_id):,} sesiones reales de LongMemEval "
        f"({LONGMEMEVAL_REPO_ID}/{LONGMEMEVAL_FILENAME})."
    )
    return sessions_by_id


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
    sessions_by_id = _fetch_raw_longmemeval_texts()
    all_session_ids = list(sessions_by_id)
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
            if session_idx >= len(all_session_ids):
                raise ValueError(
                    f"El corpus solo tiene {accumulated_tokens:,} tokens, insuficientes "
                    f"para la partición de {target:,} tokens."
                )
            session_str = sessions_by_id[all_session_ids[session_idx]]

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
                session_ids=all_session_ids[:session_idx],
            )
        )

    return partitions


def load_quality_questions(
    available_session_ids: List[str],
    max_questions: int,
    seed: int = 42,
    pilot: bool = False,
) -> List[QualityQuestion]:
    """
    Selecciona preguntas de LongMemEval cuyas sesiones de evidencia están todas dentro
    de la partición indexada, con una muestra reproducible.
    """
    available = set(available_session_ids)
    questions = [
        QualityQuestion(
            question_id=row["question_id"],
            question_type=row["question_type"],
            question=row["question"],
            answer=str(row["answer"]),
            session_ids=list(row["haystack_session_ids"]),
        )
        for row in _load_longmemeval_records()
        if set(row["haystack_session_ids"]) <= available
        and is_pilot_question(row["question_id"]) == pilot
    ]
    random.Random(seed).shuffle(questions)
    return questions[:max_questions]


def parse_longmemeval_date(date: str) -> datetime:
    """Convierte fechas de LongMemEval como '2023/04/10 (Mon) 17:50' a datetime UTC."""
    return datetime.strptime(date, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)


def load_longmemeval_temporal_cases(
    max_cases: Optional[int] = None, seed: int = 42, pilot: bool = False
) -> List[EvalCase]:
    """
    Construye un caso por cada pregunta temporal de LongMemEval, con sus sesiones. La
    fecha de la pregunta sirve de ventana (VT = TT) para la capa as-of exploratoria.
    """
    cases: List[EvalCase] = []
    for row in _load_longmemeval_records():
        if row["question_type"] not in ("temporal-reasoning", "knowledge-update"):
            continue
        if is_pilot_question(row["question_id"]) != pilot:
            continue
        episodes = [
            TemporalEpisode(
                episode_id=session_id,
                text=_format_session(date, session),
                recorded_at=parse_longmemeval_date(date),
            )
            for session_id, date, session in zip(
                row["haystack_session_ids"],
                row["haystack_dates"],
                row["haystack_sessions"],
            )
        ]
        cases.append(
            EvalCase(
                case_id=f"LME_{row['question_id']}",
                source="longmemeval",
                group_id=f"LME_{row['question_id']}",
                category=row["question_type"],
                episodes=sorted(episodes, key=lambda e: e.recorded_at),
                question=f"[Question date: {row['question_date']}] {row['question']}",
                ground_truth=str(row["answer"]),
                query_vt=parse_longmemeval_date(row["question_date"]),
                query_tt=parse_longmemeval_date(row["question_date"]),
            )
        )
    if max_cases is not None:
        cases = random.Random(seed).sample(cases, min(max_cases, len(cases)))
    return cases


# Atributos de los escenarios sintéticos: (sujetos, etiqueta del atributo, valores posibles)
_SYNTHETIC_ATTRIBUTES = [
    (
        ["Carlos Mendoza", "Priya Raman", "Tomas Novak", "Aiko Sato", "Lena Fischer"],
        "city of residence",
        ["Barcelona", "Munich", "Zurich", "Stockholm", "Singapore", "Lima", "Oslo", "Porto"],
    ),
    (
        ["Valeria Gomez", "Daniel Osei", "Marta Silva", "Ivan Petrov", "Sara Haddad"],
        "employer",
        ["Hospital Clinic", "Max Planck Institute", "ETH AI Center", "MIT CSAIL",
         "Karolinska Institute", "Charite Berlin", "Institut Pasteur", "Imperial College"],
    ),
    (
        ["Elena Rostova", "Hugo Laurent", "Nadia Karim", "Paolo Conti", "Grace Mbeki"],
        "prescribed medication",
        ["Metformin 500mg", "Sitagliptin 100mg", "Empagliflozin 10mg", "Semaglutide 1mg",
         "Liraglutide 1.2mg", "Glipizide 5mg", "Pioglitazone 15mg", "Dapagliflozin 10mg"],
    ),
    (
        ["Project Helios", "Project Orion", "Project Vega", "Project Lyra", "Project Atlas"],
        "main database",
        ["PostgreSQL", "MongoDB", "Neo4j", "FalkorDB", "Cassandra", "Redis", "MariaDB", "CockroachDB"],
    ),
    (
        ["Andes Cloud", "Vertex BioLabs", "NovaTech Robotics", "Aether Biomedical", "Zephyr Systems"],
        "annual AI budget",
        ["1.2 million USD", "2.8 million USD", "4.5 million USD", "7 million USD",
         "9.5 million USD", "12 million USD", "15 million USD", "21 million USD"],
    ),
    (
        ["Mateo Soriano", "Ingrid Larsen", "Kofi Mensah", "Yuki Tanaka", "Clara Weiss"],
        "job title",
        ["Senior Researcher", "Technical Lead", "Engineering Director", "Vice President of AI",
         "Chief AI Officer", "Principal Scientist", "Head of Research", "Staff Engineer"],
    ),
]


def load_bitemporal_evaluation_cases(
    num_scenarios: int = 30, seed: int = 42
) -> List[EvalCase]:
    """
    Genera escenarios bi-temporales SINTÉTICOS en los que VT y TT difieren de verdad:
    1. En T1 se registra el valor A.
    2. En T3 se informa tarde que el valor cambió a B en T2 (T2 < T3).
    3. En T4 se corrige lo anterior: el cambio de T2 fue en realidad a C.
    Cada escenario genera cinco preguntas con la misma pregunta base y distinta ventana.
    """
    rng = random.Random(seed)
    subjects = [
        (subject, label, values)
        for group, label, values in _SYNTHETIC_ATTRIBUTES
        for subject in group
    ]
    rng.shuffle(subjects)
    if num_scenarios > len(subjects):
        raise ValueError(f"Solo hay {len(subjects)} sujetos para los escenarios sintéticos.")

    cases: List[EvalCase] = []
    for idx, (subject, label, values) in enumerate(subjects[:num_scenarios]):
        value_a, value_b, value_c = rng.sample(values, 3)
        t1 = datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 120))
        t2 = t1 + timedelta(days=rng.randint(150, 240))
        t3 = t2 + timedelta(days=rng.randint(100, 160))
        t4 = t3 + timedelta(days=rng.randint(120, 200))
        now = t4 + timedelta(days=rng.randint(30, 90))
        # Dentro del periodo en que el cambio de T2 aún no se había informado
        between = t2 + timedelta(days=60)

        def _day(moment: datetime) -> str:
            return moment.strftime("%Y-%m-%d")

        group_id = f"SYN_{idx + 1:02d}"
        episodes = [
            TemporalEpisode(
                f"{group_id}_e1",
                f"[Date: {_day(t1)}] User: Please note that the {label} of {subject} is {value_a}.",
                t1,
            ),
            TemporalEpisode(
                f"{group_id}_e2",
                f"[Date: {_day(t3)}] User: I forgot to tell you earlier: on {_day(t2)}, "
                f"the {label} of {subject} changed to {value_b}.",
                t3,
            ),
            TemporalEpisode(
                f"{group_id}_e3",
                f"[Date: {_day(t4)}] User: Correction to what I told you before: the change on "
                f"{_day(t2)} was actually to {value_c}, not {value_b}. The {label} of {subject} "
                f"has been {value_c} since {_day(t2)}.",
                t4,
            ),
        ]
        # (categoría, tiempo de validez, tiempo de transacción, respuesta, trampas)
        windows = [
            ("current-state", now, now, value_c, [value_a, value_b]),
            ("tt-before-late-arrival", between, t3 - timedelta(days=1), value_a, [value_b, value_c]),
            ("tt-after-late-arrival", between, t3 + timedelta(days=30), value_b, [value_a, value_c]),
            ("retroactive-correction", between, now, value_c, [value_a, value_b]),
            ("pre-change-audit", t2 - timedelta(days=30), now, value_a, [value_b, value_c]),
        ]
        for category, query_vt, query_tt, answer, traps in windows:
            cases.append(
                EvalCase(
                    case_id=f"{group_id}_{category}",
                    source="sintetico",
                    group_id=group_id,
                    category=category,
                    episodes=episodes,
                    question=(
                        f"Based only on what had been recorded as of {_day(query_tt)}, "
                        f"what was the {label} of {subject} on {_day(query_vt)}?"
                    ),
                    ground_truth=answer,
                    query_vt=query_vt,
                    query_tt=query_tt,
                    anachronistic_traps=traps,
                )
            )
    return cases
