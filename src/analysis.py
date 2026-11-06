"""
Análisis estadístico de los resultados:
- Experimento 1: media y desviación estándar por partición, y calidad de respuesta.
- Experimento 2: accuracy con intervalo de Wilson, comparaciones planificadas con McNemar y
  ajuste de Holm, tasas de error, etapa en que falló cada caso sintético y calidad de la extracción.
- Muestra aleatoria de veredictos para revisar el Juez LLM a mano.
- Sensibilidad: accuracy sobre los casos que todos los sistemas evaluaron y McNemar contando
  los errores como fallos.
Cada fila indica si pertenece al análisis principal, al exploratorio o al de sensibilidad.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config import ExperimentConfig
from src.evaluation import (
    LIGHTRAG_LABEL,
    ZEP_ASOF_LABEL,
    ZEP_RAW_LABEL,
    holm_adjust,
    mcnemar_exact,
    wilson_interval,
)

JUDGE_REVIEW_SAMPLE_PER_METHOD = 30

# Comparaciones definidas antes de ver los resultados. El resto de pares es exploratorio.
PLANNED_COMPARISONS: Dict[str, List[Tuple[str, str]]] = {
    "sintetico": [
        (ZEP_ASOF_LABEL, LIGHTRAG_LABEL),
        (ZEP_ASOF_LABEL, ZEP_RAW_LABEL),
        (ZEP_RAW_LABEL, LIGHTRAG_LABEL),
    ],
    "longmemeval": [(ZEP_RAW_LABEL, LIGHTRAG_LABEL)],
    "E2GraphRAG": [("E2GraphRAG", "LightRAG")],
}


def _as_bool(series: pd.Series) -> pd.Series:
    """Normaliza veredictos leídos del CSV, donde pueden aparecer como texto."""
    return series.map(lambda v: str(v).strip().lower() == "true" or v == 1)


def _accuracy_rows(
    df: pd.DataFrame, group_columns: List[str], experiment: str, analysis: str
) -> List[Dict[str, Any]]:
    """Calcula la accuracy y su intervalo para cada método dentro de cada grupo."""
    rows: List[Dict[str, Any]] = []
    for group_key, group in df.groupby(group_columns, sort=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        for metodo, by_method in group.groupby("metodo", sort=False):
            hits, total = int(by_method["acierto"].sum()), len(by_method)
            low, high = wilson_interval(hits, total)
            row = {
                "experimento": experiment,
                "analisis": analysis,
                "grupo": " / ".join(str(k) for k in group_key),
                "metodo": metodo,
                "metrica": "accuracy",
                "n": total,
                "valor": hits / total,
                "ic95_bajo": low,
                "ic95_alto": high,
            }
            if "escenario" in by_method:
                row["n_escenarios"] = by_method["escenario"].nunique()
            rows.append(row)
    return rows


def _mcnemar_row(
    group: pd.DataFrame, first: str, second: str
) -> Dict[str, Any]:
    """Prueba de McNemar sobre los casos evaluados en ambos métodos."""
    pivot = group.pivot_table(
        index="clave_par", columns="metodo", values="acierto", aggfunc="first"
    )
    if first not in pivot or second not in pivot:
        return {}
    pairs = pivot[[first, second]].dropna().astype(bool)
    only_first = int((pairs[first] & ~pairs[second]).sum())
    only_second = int((~pairs[first] & pairs[second]).sum())
    return {
        "metodo": f"{first} vs {second}",
        "metrica": "mcnemar",
        "n": len(pairs),
        "solo_primero_acierta": only_first,
        "solo_segundo_acierta": only_second,
        "mcnemar_p": mcnemar_exact(only_first, only_second),
    }


def _comparison_rows(
    df: pd.DataFrame, source: str, experiment: str
) -> List[Dict[str, Any]]:
    """Compara todos los pares; Holm se aplica solo a las comparaciones planificadas."""
    planned = PLANNED_COMPARISONS[source]
    planned_rows = [_mcnemar_row(df, *pair) for pair in planned]
    planned_rows = [row for row in planned_rows if row]
    for row, adjusted in zip(
        planned_rows, holm_adjust([row["mcnemar_p"] for row in planned_rows])
    ):
        row.update(analisis="principal", mcnemar_p_holm=adjusted)

    other_rows = []
    for first, second in combinations(sorted(df["metodo"].unique()), 2):
        if (first, second) in planned or (second, first) in planned:
            continue
        row = _mcnemar_row(df, first, second)
        if row:
            row["analisis"] = "exploratorio"
            other_rows.append(row)

    rows = planned_rows + other_rows
    for row in rows:
        row.update(experimento=experiment, grupo=source)
    return rows


def _category_comparison_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Comparaciones por categoría: exploratorias y sin ajuste, porque n es pequeño."""
    rows = []
    for (source, category), group in df.groupby(
        ["fuente_dataset", "categoria_temporal"], sort=False
    ):
        for first, second in combinations(sorted(group["metodo"].unique()), 2):
            row = _mcnemar_row(group, first, second)
            if row:
                row.update(
                    experimento="exp2_categoria",
                    analisis="exploratorio",
                    grupo=f"{source} / {category}",
                )
                rows.append(row)
    return rows


def _coverage_rows(
    df: pd.DataFrame, group_columns: List[str], experiment: str
) -> List[Dict[str, Any]]:
    """
    Tasas de error y sin veredicto, y una accuracy de sensibilidad que cuenta esas filas
    como fallos: si un sistema falla más, su accuracy sobre lo evaluado queda inflada.
    """
    rows: List[Dict[str, Any]] = []
    for group_key, group in df.groupby(group_columns, sort=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        metodo = group_key[-1]
        label = " / ".join(str(k) for k in group_key[:-1])
        total = len(group)
        hits = int((_as_bool(group["correcto"]) & (group["juez_fuente"] == "llm")).sum())
        low, high = wilson_interval(hits, total)
        common = {"experimento": experiment, "analisis": "principal", "grupo": label,
                  "metodo": metodo, "n": total}
        rows.append(
            {**common, "metrica": "tasa_error_sistema",
             "valor": (group["juez_fuente"] == "error").mean()}
        )
        rows.append(
            {**common, "metrica": "tasa_sin_veredicto",
             "valor": (group["juez_fuente"] == "sin_veredicto").mean()}
        )
        rows.append(
            {**common, "metrica": "accuracy_errores_como_fallo", "valor": hits / total,
             "ic95_bajo": low, "ic95_alto": high}
        )
    return rows


def _sensitivity_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Comprueba si las conclusiones dependen de los casos que algún sistema no pudo evaluar:
    accuracy solo sobre los casos que todos evaluaron, y McNemar planificado contando los
    errores y la falta de veredicto como fallos.
    """
    rows: List[Dict[str, Any]] = []
    evaluated = df["juez_fuente"] == "llm"
    df = df.assign(acierto=_as_bool(df["correcto"]) & evaluated, clave_par=df["case_id"])
    for source, group in df.groupby("fuente_dataset", sort=False):
        methods_per_case = group[evaluated.loc[group.index]].groupby("case_id")["metodo"].nunique()
        common_cases = methods_per_case[methods_per_case == group["metodo"].nunique()].index
        common = group[group["case_id"].isin(common_cases)]
        accuracy = _accuracy_rows(common, ["fuente_dataset"], "exp2_sensibilidad", "sensibilidad")
        for row in accuracy:
            row["metrica"] = "accuracy_casos_comunes"
        rows += accuracy

        planned = [_mcnemar_row(group, *pair) for pair in PLANNED_COMPARISONS[source]]
        planned = [row for row in planned if row]
        for row, adjusted in zip(planned, holm_adjust([row["mcnemar_p"] for row in planned])):
            row.update(
                experimento="exp2_sensibilidad",
                analisis="sensibilidad",
                grupo=source,
                metrica="mcnemar_errores_como_fallo",
                mcnemar_p_holm=adjusted,
            )
        rows += planned
    return rows


def _failure_stage_rows(synthetic: pd.DataFrame) -> List[Dict[str, Any]]:
    """Reparte los fallos de Graphiti entre extracción, fecha, recuperación y respuesta."""
    rows: List[Dict[str, Any]] = []
    graphiti = synthetic[synthetic["metodo"] != LIGHTRAG_LABEL]
    for metodo, group in graphiti.groupby("metodo", sort=False):
        failures = group[~group["acierto"]]
        if failures.empty:
            continue
        in_graph = _as_bool(failures["oro_en_grafo"])
        in_facts = _as_bool(failures["oro_en_hechos"])
        # Solo la capa as-of descarta hechos por su fecha
        wrong_date = ~_as_bool(failures["fecha_oro_correcta"]) & (metodo == ZEP_ASOF_LABEL)
        stages = {
            "fallo_extraccion": ~in_graph,
            "fallo_fecha_extraida": in_graph & ~in_facts & wrong_date,
            "fallo_recuperacion": in_graph & ~in_facts & ~wrong_date,
            "fallo_respuesta": in_facts,
        }
        for stage, mask in stages.items():
            rows.append(
                {
                    "experimento": "exp2_diagnostico",
                    "analisis": "exploratorio",
                    "grupo": "sintetico",
                    "metodo": metodo,
                    "metrica": stage,
                    "n": len(failures),
                    "valor": mask.mean(),
                }
            )
    return rows


def _extraction_quality_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Calidad de lo que extrajo Graphiti con este LLM, para no atribuirla a la arquitectura.
    Se calcula sobre un brazo porque ambos comparten el mismo grafo.
    """
    rows: List[Dict[str, Any]] = []
    asof = df[df["metodo"] == ZEP_ASOF_LABEL]
    for source, group in asof.groupby("fuente_dataset", sort=False):
        by_group = group.groupby("escenario")["proporcion_hechos_no_latinos"].first().dropna()
        rows.append(
            {
                "experimento": "exp2_extraccion",
                "analisis": "exploratorio",
                "grupo": source,
                "metodo": "Graphiti",
                "metrica": "proporcion_hechos_no_latinos",
                "n": len(by_group),
                "valor": by_group.mean(),
            }
        )
    synthetic = asof[asof["fuente_dataset"] == "sintetico"]
    for category, group in synthetic.groupby("categoria_temporal", sort=False):
        correct = int(_as_bool(group["fecha_oro_correcta"]).sum())
        low, high = wilson_interval(correct, len(group))
        rows.append(
            {
                "experimento": "exp2_extraccion",
                "analisis": "exploratorio",
                "grupo": f"sintetico / {category}",
                "metodo": "Graphiti",
                "metrica": "tasa_fecha_oro_correcta",
                "n": len(group),
                "valor": correct / len(group),
                "ic95_bajo": low,
                "ic95_alto": high,
            }
        )
    return rows


def summarize_exp1(csv_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    rows: List[Dict[str, Any]] = []
    for (metodo, label), group in df.groupby(["metodo", "particion_label"], sort=False):
        for metric in (
            "latencia_s_por_mb",
            "wall_clock_total_s",
            "gpu_llm_total_tokens",
            "llm_llamadas_fallidas",
        ):
            rows.append(
                {
                    "experimento": "exp1_indexacion",
                    "analisis": "principal",
                    "grupo": label,
                    "metodo": metodo,
                    "metrica": metric,
                    "n": len(group),
                    "valor": group[metric].mean(),
                    "desviacion_estandar": group[metric].std(),
                }
            )
    return rows


def summarize_exp1_quality(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    rows = _coverage_rows(df, ["particion_label", "metodo"], "exp1_calidad_cobertura")
    df = df[df["juez_fuente"] == "llm"].copy()
    df["acierto"] = _as_bool(df["correcto"])
    df["clave_par"] = df["question_id"]
    rows += _accuracy_rows(df, ["particion_label"], "exp1_calidad", "principal")
    rows += _comparison_rows(df, "E2GraphRAG", "exp1_calidad")
    return rows


def summarize_exp2(csv_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    rows = _coverage_rows(df, ["fuente_dataset", "metodo"], "exp2_cobertura")
    rows += _sensitivity_rows(df)

    df = df[df["juez_fuente"] == "llm"].copy()
    df["acierto"] = _as_bool(df["correcto"])
    df["clave_par"] = df["case_id"]

    overall = _accuracy_rows(df, ["fuente_dataset"], "exp2", "principal")
    # En LongMemEval, la capa as-of no estaba en la hipótesis principal
    for row in overall:
        if row["grupo"] == "longmemeval" and row["metodo"] == ZEP_ASOF_LABEL:
            row["analisis"] = "exploratorio"
    rows += overall
    rows += _accuracy_rows(
        df, ["fuente_dataset", "categoria_temporal"], "exp2_categoria", "exploratorio"
    )
    for source, group in df.groupby("fuente_dataset", sort=False):
        rows += _comparison_rows(group, source, "exp2")
    rows += _category_comparison_rows(df)
    rows += _extraction_quality_rows(df)

    synthetic = df[df["fuente_dataset"] == "sintetico"].copy()
    if len(synthetic):
        rows += _failure_stage_rows(synthetic)
        synthetic["acierto"] = _as_bool(synthetic["error_anacronico"])
        anachronistic = _accuracy_rows(
            synthetic, ["fuente_dataset"], "exp2_anacronismo", "principal"
        )
        for row in anachronistic:
            row["metrica"] = "tasa_error_anacronico"
        rows += anachronistic
    return rows


def write_judge_review_sample(csv_path: Path, output_path: Path) -> None:
    """Exporta una muestra reproducible para anotar a mano y medir el acuerdo con el juez."""
    df = pd.read_csv(csv_path)
    df = df[df["juez_fuente"] == "llm"]
    samples = [
        group.sample(min(JUDGE_REVIEW_SAMPLE_PER_METHOD, len(group)), random_state=42)
        for _, group in df.groupby("metodo")
    ]
    sample = pd.concat(samples).sample(frac=1, random_state=42)
    # El método se omite para que la revisión manual sea ciega; `fila` permite recuperarlo
    review = sample[["pregunta", "ground_truth", "respuesta_generada", "correcto"]].rename(
        columns={"correcto": "veredicto_juez"}
    )
    review.insert(0, "fila", sample.index)
    review["veredicto_manual"] = ""
    review.to_csv(output_path, index=False, encoding="utf-8")


def run_analysis(config: ExperimentConfig | None = None) -> pd.DataFrame:
    """Genera el resumen estadístico y la muestra para revisar el juez."""
    cfg = config or ExperimentConfig()
    exp2_csv = cfg.base_dir / cfg.exp2_csv_path
    rows = (
        summarize_exp1(cfg.base_dir / cfg.exp1_csv_path)
        + summarize_exp1_quality(cfg.base_dir / cfg.exp1_quality_csv_path)
        + summarize_exp2(exp2_csv)
    )
    df = pd.DataFrame(rows)
    stats_path = cfg.base_dir / cfg.stats_csv_path
    df.to_csv(stats_path, index=False, encoding="utf-8")
    review_path = cfg.base_dir / cfg.judge_review_csv_path
    write_judge_review_sample(exp2_csv, review_path)
    cfg.sync_to_google_drive(stats_path)
    cfg.sync_to_google_drive(review_path)
    print(f"[OK] Resumen estadístico guardado en: {stats_path.resolve()}")
    print(f"[OK] Muestra para revisión manual del juez: {review_path.resolve()}")
    return df


if __name__ == "__main__":
    run_analysis()
