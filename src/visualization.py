"""
BLOQUE 4: Visualización y Reporte de Resultados (Publication-Ready, 300 DPI).
Genera dos figuras académicas a partir de 'experimento1_resultados.csv' y
'experimento2_resultados.csv' usando Matplotlib y Seaborn:
1. Figura 1: Latencia de Indexación (s/MB) y Consumo de Tokens GPU (E2GraphRAG vs LightRAG).
2. Figura 2: Recall Temporal y % de Errores Anacrónicos (Zep/Graphiti vs LightRAG).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ExperimentConfig


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


def plot_experiment1_indexing_latency(
    csv_path: str | Path = "experimento1_resultados.csv",
    output_path: str | Path = "fig1_latencia_indexacion.png",
) -> Path:
    """
    Genera el Gráfico 1 (Publication-Ready):
    - Subgráfico Izquierdo (Barras + Línea): Latencia de Indexación (s/MB) E2GraphRAG vs LightRAG.
    - Subgráfico Derecho (Líneas): Escalabilidad en Consumo de Tokens LLM en GPU.
    """
    configure_publication_style()
    df = pd.read_csv(csv_path)

    palette = {
        "E2GraphRAG": "#1b9e77",  # Verde esmeralda académico
        "LightRAG": "#d95f02",    # Naranja intenso
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- Panel A: Barras + Líneas de Latencia de Indexación (s/MB) ---
    sns.barplot(
        data=df,
        x="particion_label",
        y="latencia_s_por_mb",
        hue="metodo",
        palette=palette,
        edgecolor="#222222",
        alpha=0.88,
        ax=ax1,
    )

    # Superponer líneas de tendencia sobre los centros de cada grupo
    partitions = df["particion_label"].unique().tolist()
    x_coords = np.arange(len(partitions))
    for offset, metodo in [(-0.2, "E2GraphRAG"), (0.2, "LightRAG")]:
        sub = (
            df[df["metodo"] == metodo]
            .set_index("particion_label")
            .reindex(partitions)
        )
        if not sub["latencia_s_por_mb"].isna().all():
            ax1.plot(
                x_coords + offset,
                sub["latencia_s_por_mb"].values,
                marker="o",
                color="#111111",
                linewidth=1.4,
                markersize=5,
                linestyle=":",
            )

    # Anotar valores sobre las barras
    for container in ax1.containers:
        ax1.bar_label(
            container,
            fmt="%.1f s/MB",
            padding=3,
            fontsize=8.5,
            fontweight="bold",
        )

    ax1.set_title(
        "(a) Latencia de Muro en Indexación (s/MB)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlabel("Tamaño de Partición LongMemEval (Tokens)", fontweight="bold")
    ax1.set_ylabel("Latencia de Indexación (segundos / MB) ↓", fontweight="bold")
    ax1.legend(title="Arquitectura", loc="upper left")

    # --- Panel B: Consumo de Tokens de GPU durante Indexación ---
    df["gpu_tokens_miles"] = df["gpu_llm_total_tokens"] / 1_000.0
    sns.lineplot(
        data=df,
        x="particion_label",
        y="gpu_tokens_miles",
        hue="metodo",
        style="metodo",
        palette=palette,
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=9,
        ax=ax2,
    )

    for _, row in df.iterrows():
        ax2.annotate(
            f"{row['gpu_tokens_miles']:.1f}k",
            (row["particion_label"], row["gpu_tokens_miles"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax2.set_title(
        "(b) Consumo de Tokens LLM en GPU por Partición",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax2.set_xlabel("Tamaño de Partición LongMemEval (Tokens)", fontweight="bold")
    ax2.set_ylabel("Tokens LLM Consumidos en GPU (Miles) ↓", fontweight="bold")
    ax2.legend(title="Arquitectura", loc="upper left")

    fig.suptitle(
        "Experimento 1 (Problema P1): Eficiencia en Indexación — E2GraphRAG vs. LightRAG",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out_file = Path(output_path)
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    fig.savefig(out_file.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico 1 guardado en: {out_file.resolve()}")
    return out_file


def plot_experiment2_bitemporal_metrics(
    csv_path: str | Path = "experimento2_resultados.csv",
    output_path: str | Path = "fig2_bitemporal_recall_errores.png",
) -> Path:
    """
    Genera el Gráfico 2 (Publication-Ready):
    - Subgráfico Izquierdo: Comparación Global de Recall Temporal (0-100%) y
      % de Errores Anacrónicos entre Zep/Graphiti y LightRAG.
    - Subgráfico Derecho: Recall Temporal desglosado por tipo de consulta bi-temporal.
    """
    configure_publication_style()
    df = pd.read_csv(csv_path)

    agg = (
        df.groupby("metodo", as_index=False)
        .agg(
            Recall_Temporal_Pct=(
                "recall_temporal",
                lambda s: float(s.mean()) * 100.0,
            ),
            Errores_Anacronicos_Pct=(
                "error_anacronico",
                lambda s: float(s.mean()) * 100.0,
            ),
        )
    )

    melted = agg.melt(
        id_vars="metodo",
        value_vars=["Recall_Temporal_Pct", "Errores_Anacronicos_Pct"],
        var_name="Metrica",
        value_name="Porcentaje",
    )
    melted["Metrica"] = melted["Metrica"].replace(
        {
            "Recall_Temporal_Pct": "Recall Temporal (%) ↑",
            "Errores_Anacronicos_Pct": "Tasa de Errores Anacrónicos (%) ↓",
        }
    )

    palette_methods = {
        "Zep / Graphiti (Bi-temporal)": "#2b5c8f",  # Azul zafiro
        "LightRAG (Estático)": "#c53a32",           # Rojo carmesí
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- Panel A: Recall Temporal vs % Errores Anacrónicos ---
    sns.barplot(
        data=melted,
        x="Metrica",
        y="Porcentaje",
        hue="metodo",
        palette=palette_methods,
        edgecolor="#222222",
        alpha=0.90,
        ax=ax1,
    )

    for container in ax1.containers:
        ax1.bar_label(
            container,
            fmt="%.1f%%",
            padding=3,
            fontsize=9.5,
            fontweight="bold",
        )

    ax1.set_ylim(0, 115)
    ax1.set_title(
        "(a) Desempeño Global: Recall Temporal vs. Errores Anacrónicos",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Porcentaje (%) [Escala 0 - 100]", fontweight="bold")
    ax1.legend(title="Sistema Evaluado", loc="upper right")

    # --- Panel B: Recall Temporal (0 a 1) por Categoría de Ventana Bi-temporal ---
    sns.barplot(
        data=df,
        x="categoria_temporal",
        y="recall_temporal",
        hue="metodo",
        palette=palette_methods,
        edgecolor="#222222",
        errorbar=None,
        alpha=0.90,
        ax=ax2,
    )

    for container in ax2.containers:
        ax2.bar_label(
            container,
            fmt="%.2f",
            padding=3,
            fontsize=9.0,
            fontweight="bold",
        )

    ax2.set_ylim(0, 1.15)
    ax2.set_title(
        "(b) Recall Temporal (0 a 1) por Tipo de Consulta (t_VT, t_TT)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax2.set_xlabel("Categoría de Consulta Bi-temporal", fontweight="bold")
    ax2.set_ylabel("Recall Temporal Promedio (0.0 - 1.0) ↑", fontweight="bold")
    ax2.legend(title="Sistema Evaluado", loc="upper right")

    fig.suptitle(
        "Experimento 2 (Problema P2): Consistencia Bi-temporal — Zep / Graphiti vs. LightRAG",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out_file = Path(output_path)
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    fig.savefig(out_file.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
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
