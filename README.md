# Arquitectura Experimental RAG con Grafos de Conocimiento (PFC)

Repositorio oficial de código para los experimentos del Trabajo de Fin de Carrera (PFC) sobre **Eficiencia de Indexación (P1)** y **Consistencia Bi-temporal (P2)** en sistemas *Graph-based Retrieval-Augmented Generation (GraphRAG)*.

Diseñado para ejecutarse con **$0 de costo en APIs** en **Google Colab Pro / Google AI Pro** (GPU NVIDIA A100 / L4 + CPU High-RAM).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcelo130102/pfc1/blob/main/notebooks/PFC_Experimentos_RAG_Colab.ipynb)

---

## 1. Componentes Evaluados

| Rol | Arquitectura | Mecanismo Clave | Problema que Aborda |
| :--- | :--- | :--- | :--- |
| **Baseline de Control** | **LightRAG** (`lightrag-hku`) | Extracción generativa LLM de entidades/relaciones + grafo estático | Línea base estándar en GraphRAG |
| **Técnica 1 (P1)** | **E²GraphRAG** ([portado del repo oficial](https://github.com/YiboZhao624/E-2GraphRAG), MIT, en `src/e2graphrag.py`) | Extracción NER en CPU (`SpaCy`) + Grafo de Coocurrencia Oracional (`NetworkX`) + Árbol Jerárquico de Resúmenes en GPU | Cuello de botella computacional y alto consumo de tokens GPU durante la indexación |
| **Técnica 2 (P2)** | **Zep / Graphiti** (`graphiti-core`) | Modelo Bi-temporal $(t_{VT} \times t_{TT})$ con intervalos de validez `[valid_at, invalid_at)` e invalidación dinámica de aristas contradictorias | Alucinaciones cronológicas y mezcla de hechos obsoletos ante actualizaciones de conocimiento |
| **Dataset** | **LongMemEval** (`xiaowu0162/longmemeval-cleaned`) | Particiones de `100k` a `2.5M` tokens y preguntas `temporal-reasoning` y `knowledge-update`, más 30 escenarios bi-temporales **sintéticos** con hechos tardíos y correcciones retroactivas | Evaluación de escalabilidad y razonamiento temporal |
| **Motor Local** | **vLLM / Ollama + BGE-M3** | `Qwen/Qwen2.5-7B-Instruct` en `localhost:8000` + `BAAI/bge-m3` | Inferencia reproducible sin costo de API |

---

## 2. Estructura del Proyecto

```text
pfc1/
├── docs/
│   └── COLAB_PRO_CONFIG.md              # Documentación interna: Guía de configuración en Google Colab Pro (A100/L4 + High-RAM)
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
    ├── e2graphrag.py                    # Indexación y recuperación de E²GraphRAG, portadas del repo oficial
    ├── evaluation.py                    # Juez LLM, intervalo de Wilson y prueba de McNemar
    ├── analysis.py                      # Resumen estadístico y muestra para revisar el juez a mano
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

## 4. Ejecución desde la CLI de Colab

```bash
colab new -s pfc --gpu A100 --high-mem
colab drivemount -s pfc
python3 scripts/colab_smoke_test.py -s pfc     # Debe terminar en SMOKE:PASS
python3 scripts/colab_launch.py -s pfc --pilot # Piloto para ajustar (archivos piloto_*)
python3 scripts/colab_launch.py -s pfc --fresh # Corrida final, con el diseño ya en un commit
colab stop -s pfc
```

Si la VM se pierde a mitad de camino, `python3 scripts/colab_launch.py -s pfc` reanuda desde los checkpoints de Drive. El piloto usa otra semilla y cerca del 10% de las preguntas, reservadas: la corrida final no las evalúa, así que ajustar con el piloto no contamina los resultados. Cada corrida guarda en `run_manifest.json` el modelo, el dataset y los hashes del código. Los resultados válidos se versionan en `results/`.

---

## 5. Documentación Interna y Ejecución en Google Colab Pro

Consulta la documentación técnica interna en [`docs/COLAB_PRO_CONFIG.md`](docs/COLAB_PRO_CONFIG.md) para los detalles de configuración de hardware **A100 + High-RAM**, asignación de VRAM y persistencia de datos. Puedes abrir directamente el cuaderno de experimentación en [`notebooks/PFC_Experimentos_RAG_Colab.ipynb`](notebooks/PFC_Experimentos_RAG_Colab.ipynb).

