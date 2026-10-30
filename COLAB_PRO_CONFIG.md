# Guía de Configuración Máxima en Google Colab Pro / Google AI Pro

Dado que cuentas con una suscripción **Google AI Pro / Google Colab Pro**, esta guía detalla cómo exprimir al máximo el hardware disponible (**NVIDIA A100-SXM4-40GB/80GB** o **NVIDIA L4-24GB** con **High-RAM**) para ejecutar los experimentos del Trabajo de Fin de Carrera con **$0 de costo en APIs externas**.

---

## 1. Selección del Entorno de Ejecución (Runtime)

1. Abre el notebook [`notebooks/PFC_Experimentos_RAG_Colab.ipynb`](notebooks/PFC_Experimentos_RAG_Colab.ipynb) en Google Colab.
2. En el menú superior, navega a:
   **Entorno de ejecución (Runtime) $\rightarrow$ Cambiar tipo de entorno de ejecución (Change runtime type)**.
3. Configura los siguientes parámetros al máximo nivel:
   - **Tipo de entorno de ejecución**: `Python 3` (Colab usa Python 3.10/3.11 por defecto).
   - **Acelerador de hardware**:
     - **Opción Recomendada (Máximo Rendimiento)**: **`A100 GPU`** (40 GB o 80 GB VRAM, ancho de banda de memoria de 1,555 GB/s y soporte nativo para FlashAttention-2 en `bfloat16`).
     - **Opción Alternativa (Eficiencia de Unidades de Cómputo)**: **`L4 GPU`** (24 GB VRAM Ada Lovelace, excelente soporte FP16/BF16).
   - **Forma del entorno (Runtime shape)**: Activa el interruptor **Alta capacidad de RAM (High-RAM)**.
     - En **A100 + High-RAM**, dispondrás de **~83.5 GB de RAM de sistema** y **12 vCPUs Intel Xeon**, lo cual es ideal para paralelizar la extracción de entidades de **E2GraphRAG con SpaCy (`n_process=8`)** sin cuellos de botella en CPU.

---

## 2. Configuración de Secretos en Colab (`HF_TOKEN`)

Para descargar modelos con licencia comunitaria como `meta-llama/Llama-3.1-8B-Instruct` desde HuggingFace:

1. Haz clic en el ícono de la **Llave (Secrets)** en la barra lateral izquierda de Google Colab.
2. Añade un nuevo secreto:
   - **Nombre**: `HF_TOKEN`
   - **Valor**: Tu token de lectura de HuggingFace (`hf_...`).
3. Activa el interruptor **Acceso al cuaderno (Notebook access)**.

> **Nota**: Si no deseas usar un token con restricciones de licencia (gated repo), el código soporta cambiar con un solo parámetro a **`Qwen/Qwen2.5-7B-Instruct`**, el cual es 100% abierto (Apache 2.0) y no requiere aprobación previa en HuggingFace.

---

## 3. Distribución Óptima de VRAM (A100 vs L4)

En nuestros experimentos coexisten en la misma GPU el servidor **vLLM** (inferencia generativa + Juez LLM) y el modelo de embeddings densos **`BAAI/bge-m3`** (~2.2 GB en FP16).

| Componente en GPU | Perfil **NVIDIA A100 (40 GB)** | Perfil **NVIDIA L4 (24 GB)** |
| :--- | :--- | :--- |
| **Modelo Base LLM** | `meta-llama/Llama-3.1-8B-Instruct` (BF16, ~16 GB) | `meta-llama/Llama-3.1-8B-Instruct` (BF16, ~16 GB) |
| **`--gpu-memory-utilization` (vLLM)** | **`0.80`** (~32 GB para pesos + KV Cache masivo) | **`0.72`** (~17.2 GB para pesos + KV Cache) |
| **`--max-model-len` (Contexto)** | **`32768`** tokens | **`16384`** tokens |
| **`--max-num-seqs` (Concurrencia)** | **`64`** peticiones concurrentes (acelera LightRAG y Árbol E2) | **`16`** peticiones concurrentes |
| **Embeddings (`BAAI/bge-m3`)** | GPU `cuda:0` en `float16` (`batch_size=128`) | GPU `cuda:0` en `float16` (`batch_size=32`) |
| **SpaCy NER + Coocurrencia (E2GraphRAG)** | CPU Multiproceso (`n_process=8`, usa ~12 GB de los 83 GB de RAM) | CPU Multiproceso (`n_process=4`) |

El módulo [`src/config.py`](src/config.py) detecta automáticamente si estás corriendo sobre una **A100** o una **L4** mediante `torch.cuda.get_device_name(0)` y ajusta estos parámetros sin intervención manual.

---

## 4. Persistencia en Google Drive (Evitar Pérdida de Datos por Desconexión)

El **Experimento 1** en la partición de `1M tokens` de **LightRAG** realiza miles de llamadas al LLM para extraer entidades y relaciones, lo cual puede tardar varios minutos incluso en una A100.

Para proteger tus resultados ante posibles desconexiones de sesión:
1. Monta Google Drive al inicio del cuaderno:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. El código guarda incrementalmente cada punto de corte (`100k`, `250k`, `500k`, `1M`) en `experimento1_resultados.csv` y `experimento2_resultados.csv`, además de sincronizarlos con `/content/drive/MyDrive/PFC_RAG_Resultados/` si Drive está montado.

---

## 5. Ejecución del Servidor vLLM en Segundo Plano en Colab

En Google Colab, un proceso bloqueante impide ejecutar las celdas siguientes. Por ello, el **Bloque 1** lanza `vllm.entrypoints.openai.api_server` como un subproceso asíncrono en segundo plano (`subprocess.Popen`) con:
- `--enable-prefix-caching`: Reutiliza los bloques de atención (KV Cache) del *system prompt* de extracción de entidades de LightRAG y del Juez LLM, reduciendo la latencia hasta un **38%**.
- `--disable-log-requests`: Evita saturar la salida estándar de Colab durante miles de llamadas de indexación.
- Sondeo automático (*health check*) contra `http://localhost:8000/v1/models` antes de pasar al **Bloque 2**.

---

## 6. Flujo de Trabajo Recomendado con Ramas Git en Colab

Puedes clonar y cambiar entre las ramas del repositorio directamente desde una celda de Colab:

```bash
# Clonar el repositorio
!git clone <URL_DE_TU_REPOSITORIO> /content/pfc1
%cd /content/pfc1

# Ver todas las ramas de experimentación disponibles
!git branch -a

# Ejecutar el pipeline completo desde la rama principal o develop
!python -m src.exp1_indexing
!python -m src.exp2_bitemporal
!python -m src.visualization
```
