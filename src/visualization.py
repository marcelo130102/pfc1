"""
BLOQUE 4: Visualización y Reporte de Resultados (Publication-Ready, 300 DPI).
Genera dos figuras académicas a partir de 'experimento1_resultados.csv' y
'experimento2_resultados.csv' usando Matplotlib y Seaborn:
1. Figura 1: Latencia de Indexación (s/MB) y Tokens LLM (E2GraphRAG vs LightRAG), con la
   media y la desviación estándar entre corridas.
2. Figura 2: Accuracy en LongMemEval y en los escenarios bi-temporales sintéticos, y Tasa de
   Errores Anacrónicos, con intervalos de confianza de Wilson al 95%. La capa as-of en
   LongMemEval es exploratoria y se reporta solo en 'resumen_estadistico.csv'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ExperimentConfig
from src.evaluation import (
    LIGHTRAG_LABEL,
    ZEP_ASOF_LABEL,
    ZEP_RAW_LABEL,
    wilson_interval,
)

METHOD_COLORS_EXP1 = {"E2GraphRAG": "#1b9e77", "LightRAG": "#d95f02"}
METHOD_COLORS_EXP2 = {
    ZEP_ASOF_LABEL: "#2b5c8f",
    ZEP_RAW_LABEL: "#7fb0dc",
    LIGHTRAG_LABEL: "#c53a32",
}


def configure_publication_style() -> None:
    """Configura el estilo visual académico (IEEE / ACM) en Matplotlib y Seaborn."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    out_file = Path(output_path)
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    fig.savefig(out_file.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_file


def _shared_legend(fig: plt.Figure, ax: plt.Axes, title: str) -> None:
    """Coloca la leyenda bajo los paneles para que no tape ningún dato."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title=title, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.1, 1, 1))


def _plot_mean_std(
    ax: plt.Axes, df: pd.DataFrame, metric: str, order: list, scale: float = 1.0
) -> None:
    """Dibuja la media entre corridas de cada método, con su desviación estándar."""
    x_positions = np.arange(len(order))
    for metodo, color in METHOD_COLORS_EXP1.items():
        stats = (
            df[df["metodo"] == metodo]
            .groupby("particion_label")[metric]
            .agg(["mean", "std"])
            .reindex(order)
        )
        ax.errorbar(
            x_positions,
            stats["mean"] / scale,
            yerr=stats["std"].fillna(0) / scale,
            label=metodo,
            color=color,
            marker="o",
            linewidth=2.2,
            capsize=4,
        )
    ax.set_xticks(x_positions, order)


def plot_experiment1_indexing_latency(
    csv_path: str | Path = "experimento1_resultados.csv",
    output_path: str | Path = "fig1_latencia_indexacion.png",
) -> Path:
    """Genera la Figura 1: latencia (s/MB) y tokens LLM por partición."""
    configure_publication_style()
    df = pd.read_csv(csv_path)
    order = (
        df.sort_values("target_tokens")["particion_label"].drop_duplicates().tolist()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    _plot_mean_std(ax1, df, "latencia_s_por_mb", order)
    ax1.set_ylim(bottom=0)
    ax1.set_title("(a) Latencia de indexación", fontweight="bold")
    ax1.set_xlabel("Tamaño de partición de LongMemEval (tokens)", fontweight="bold")
    ax1.set_ylabel("Segundos por MB (menor es mejor)", fontweight="bold")

    _plot_mean_std(ax2, df, "gpu_llm_total_tokens", order, scale=1_000_000.0)
    ax2.set_ylim(bottom=0)
    ax2.set_title("(b) Tokens LLM consumidos", fontweight="bold")
    ax2.set_xlabel("Tamaño de partición de LongMemEval (tokens)", fontweight="bold")
    ax2.set_ylabel("Millones de tokens (menor es mejor)", fontweight="bold")

    _shared_legend(fig, ax1, "Arquitectura")
    out_file = _save_figure(fig, output_path)
    print(f"[OK] Gráfico 1 guardado en: {out_file.resolve()}")
    return out_file


def _plot_rate_bars(ax: plt.Axes, rates: pd.DataFrame, colors: Dict[str, str]) -> None:
    """Dibuja barras agrupadas de una proporción, con su intervalo de Wilson."""
    categories = rates["grupo"].drop_duplicates().tolist()
    methods = list(colors)
    width = 0.8 / len(methods)
    for idx, metodo in enumerate(methods):
        sub = rates[rates["metodo"] == metodo].set_index("grupo").reindex(categories)
        positions = np.arange(len(categories)) + (idx - (len(methods) - 1) / 2) * width
        ax.bar(
            positions,
            sub["valor"],
            width=width,
            color=colors[metodo],
            edgecolor="#222222",
            label=metodo,
            yerr=[sub["valor"] - sub["bajo"], sub["alto"] - sub["valor"]],
            capsize=3,
        )
    ax.set_xticks(np.arange(len(categories)), categories, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)


def _rates_by(df: pd.DataFrame, group_column: str, value_column: str) -> pd.DataFrame:
    rows = []
    for (group, metodo), sub in df.groupby([group_column, "metodo"], sort=False):
        hits = int(sub[value_column].map(lambda v: str(v).strip().lower() == "true").sum())
        low, high = wilson_interval(hits, len(sub))
        rows.append(
            {"grupo": group, "metodo": metodo, "valor": hits / len(sub), "bajo": low, "alto": high}
        )
    return pd.DataFrame(rows)


def plot_experiment2_bitemporal_metrics(
    csv_path: str | Path = "experimento2_resultados.csv",
    output_path: str | Path = "fig2_bitemporal_recall_errores.png",
) -> Path:
    """Genera la Figura 2: accuracy por tipo de consulta y errores anacrónicos."""
    configure_publication_style()
    df = pd.read_csv(csv_path)
    df = df[df["juez_fuente"] == "llm"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16.0, 6.0), gridspec_kw={"width_ratios": [2, 5, 5]}
    )

    longmemeval = df[df["fuente_dataset"] == "longmemeval"]
    principal_colors = {
        metodo: color for metodo, color in METHOD_COLORS_EXP2.items() if metodo != ZEP_ASOF_LABEL
    }
    _plot_rate_bars(ax1, _rates_by(longmemeval, "categoria_temporal", "correcto"), principal_colors)
    ax1.set_title("(a) LongMemEval: accuracy", fontweight="bold")
    ax1.set_ylabel("Accuracy (mayor es mejor)", fontweight="bold")

    synthetic = df[df["fuente_dataset"] == "sintetico"]
    _plot_rate_bars(ax2, _rates_by(synthetic, "categoria_temporal", "correcto"), METHOD_COLORS_EXP2)
    ax2.set_title("(b) Escenarios sintéticos: accuracy", fontweight="bold")
    ax2.set_ylabel("Accuracy (mayor es mejor)", fontweight="bold")

    _plot_rate_bars(
        ax3, _rates_by(synthetic, "categoria_temporal", "error_anacronico"), METHOD_COLORS_EXP2
    )
    ax3.set_title("(c) Escenarios sintéticos: errores anacrónicos", fontweight="bold")
    ax3.set_ylabel("Tasa de error (menor es mejor)", fontweight="bold")

    _shared_legend(fig, ax2, "Sistema evaluado")
    out_file = _save_figure(fig, output_path)
    print(f"[OK] Gráfico 2 guardado en: {out_file.resolve()}")
    return out_file


def generate_all_figures(
    config: Optional[ExperimentConfig] = None,
) -> Tuple[Path, Path]:
    """Genera las dos figuras publication-ready a partir de los archivos CSV."""
    cfg = config or ExperimentConfig()
    f1 = plot_experiment1_indexing_latency(
        csv_path=cfg.base_dir / cfg.exp1_csv_path,
        output_path=cfg.base_dir / cfg.fig1_path,
    )
    f2 = plot_experiment2_bitemporal_metrics(
        csv_path=cfg.base_dir / cfg.exp2_csv_path,
        output_path=cfg.base_dir / cfg.fig2_path,
    )
    cfg.sync_to_google_drive(f1)
    cfg.sync_to_google_drive(f1.with_suffix(".pdf"))
    cfg.sync_to_google_drive(f2)
    cfg.sync_to_google_drive(f2.with_suffix(".pdf"))
    return f1, f2


if __name__ == "__main__":
    generate_all_figures()
