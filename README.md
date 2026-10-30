# Arquitectura Experimental RAG con Grafos de Conocimiento (PFC)

Repositorio oficial de código para los experimentos del Trabajo de Fin de Carrera (PFC) sobre **Eficiencia de Indexación (P1)** y **Consistencia Bi-temporal (P2)** en sistemas *Graph-based Retrieval-Augmented Generation (GraphRAG)*.

Diseñado para ejecutarse con **$0 de costo en APIs** en **Google Colab Pro / Google AI Pro** (GPU NVIDIA A100 / L4 + CPU High-RAM).

---

## 1. Componentes Evaluados

| Rol | Arquitectura | Mecanismo Clave | Problema que Aborda |
| :--- | :--- | :--- | :--- |
| **Baseline de Control** | **LightRAG** (`lightrag-hku`) | Extracción generativa LLM de entidades/relaciones + grafo estático | Línea base estándar en GraphRAG |
| **Técnica 1 (P1)** | **E2GraphRAG** | Extracción NER en CPU (`SpaCy`) + Grafo de Coocurrencia Oracional (`NetworkX`) + Árbol Jerárquico de Resúmenes en GPU | Cuello de botella computacional y alto consumo de tokens GPU durante la indexación |
| **Técnica 2 (P2)** | **Zep / Graphiti** (`graphiti-core`) | Modelo Bi-temporal $(t_{VT} \times t_{TT})$ con intervalos de validez `[valid_at, invalid_at)` e invalidación dinámica de aristas contradictorias | Alucinaciones cronológicas y mezcla de hechos obsoletos ante actualizaciones de conocimiento |
| **Dataset** | **LongMemEval** | Particiones escalonadas (`100k`, `250k`, `500k`, `1M` tokens) y diálogos multi-sesión con evolución temporal y respuestas oro | Evaluación rigurosa de escalabilidad y razonamiento temporal |
| **Motor Local** | **vLLM / Ollama + BGE-M3** | `meta-llama/Llama-3.1-8B-Instruct` (o `Qwen/Qwen2.5-7B-Instruct`) en `localhost:8000` + `BAAI/bge-m3` | Inferencia reproducible sin costo de API |

---

## 2. Estructura del Proyecto

```text
pfc1/
├── COLAB_PRO_CONFIG.md                  # Guía de configuración máxima en Google Colab Pro (A100/L4 + High-RAM)
├── README.md                            # Documentación principal y arquitectura del repositorio
├── requirements.txt                     # Dependencias Python 3.10+
├── notebooks/
│   └── PFC_Experimentos_RAG_Colab.ipynb # Cuaderno Colab autocontenido con los 4 Bloques ejecutables celda a celda
└── src/
    ├── __init__.py
    ├── config.py                        # Detección automática de hardware (A100/L4) e hiperparámetros
    ├── inference_setup.py               # BLOQUE 1: Servidor vLLM/Ollama local, BGE-M3 y SpaCy en CPU
    ├── data_loader.py                   # Carga y particionamiento de LongMemEval (100k-1M tokens y sesiones temporales)
    ├── exp1_indexing.py                 # BLOQUE 2: Experimento 1 (E2GraphRAG vs LightRAG: s/MB y tokens GPU)
    ├── exp2_bitemporal.py               # BLOQUE 3: Experimento 2 (Zep/Graphiti vs LightRAG + LLM-as-a-Judge)
    └── visualization.py                 # BLOQUE 4: Gráficos publication-ready (300 DPI) con Matplotlib/Seaborn
```

---

## 3. Estructura de Ramas Git

El repositorio está organizado siguiendo un flujo Git modular por componente experimental:

- `main`: Rama estable con todos los bloques integrados, documentación y cuaderno de Google Colab.
- `develop`: Rama de integración continua de los módulos experimentales y visualización.
- `infra/colab-vllm-setup`: **Bloque 1** — Configuración del motor de inferencia local (`vLLM`/`Ollama`), embeddings `BAAI/bge-m3`, `SpaCy` y cargador de `LongMemEval`.
- `exp/p1-e2graphrag-indexacion`: **Bloque 2** — Pipeline del Experimento 1 (Eficiencia de Indexación: E2GraphRAG vs LightRAG).
- `exp/p2-zep-graphiti-bitemporal`: **Bloque 3** — Pipeline del Experimento 2 (Recuperación Bi-temporal: Zep/Graphiti vs LightRAG + Juez LLM).

---

## 4. Inicio Rápido en Google Colab Pro

Consulta [`COLAB_PRO_CONFIG.md`](COLAB_PRO_CONFIG.md) para los detalles de configuración de hardware **A100 + High-RAM** y abre directamente [`notebooks/PFC_Experimentos_RAG_Colab.ipynb`](notebooks/PFC_Experimentos_RAG_Colab.ipynb).
