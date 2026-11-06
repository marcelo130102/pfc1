# Estado de los experimentos

Actualizado: 2026-11-06. Sirve para retomar el trabajo en una conversación nueva.

## Situación actual

- **Aún no hay resultados válidos.** Los CSV y figuras de la corrida del 2026-11-05 01:06 son inválidos: el LLM falló en silencio y el corpus era sintético. No usarlos en el paper.
- **Smoke test en A100: `SMOKE:PASS`, pero con un problema en Graphiti.** Con Qwen, los hechos salieron en ruso y chino, y `valid_at` tomó la fecha de registro en vez de la fecha indicada en el mensaje. Por eso falló 2 de 5 preguntas as-of en el único escenario probado. La invalidación sí funcionó.
- **Decisión del autor (opción B):** Graphiti se mide con su configuración por defecto, sin instrucciones de extracción propias. Se añadieron mediciones de fecha e idioma que no cambian la evaluación.
- **Costo:** la A100 cuesta 8.88 unidades/hora. El smoke test costó 1.57 unidades (unos 11 minutos).
- **La L4 no sirve con esta configuración:** Qwen ocupa 14.29 GiB y no queda memoria suficiente para su caché.

## Piloto del 2026-11-05 (10:58 a 13:12, unas 19.3 unidades)

- **Datos reales:** usó las 940 sesiones de `longmemeval_oracle.json`. Los resultados están en `piloto_*` (Drive y raíz del repo).
- **Experimento 1:** sin llamadas fallidas. LightRAG tardó 874 s/MB en 100k y 936 s/MB en 250k, unas 11 veces más que E²GraphRAG, porque extrae con el LLM (9.4 veces más tokens).
- **Bucles de repetición de Qwen** con `repetition_penalty` 1.05, que vLLM ya aplicaba desde `generation_config.json`:
  - LightRAG: 2188 avisos de relaciones con campos faltantes, que se descartan. `llm_llamadas_fallidas` no los cuenta.
  - Graphiti: 3/10 casos de LongMemEval con `JSONDecodeError`. Esos 3 grupos consumieron 45 de los 77 minutos del Experimento 2.
- **Causa del costo en Graphiti:** `OpenAIGenericClient` ignoraba el `max_tokens=4096` y pedía 16384. Además, reintentaba 4 veces cada JSON cortado, con temperatura 0, así que repetía el mismo bucle.
- **Errores del cierre:** el pipeline falló al generar las figuras (`yerr` negativo por redondeo en Wilson), así que el manifiesto quedó `en_curso`. En el Mac, el lanzador se detuvo porque Colab pidió reautorizar la CLI.
- **Accuracy del piloto:** solo sirve para depurar, no para ajustar el diseño. En los escenarios sintéticos: Graphiti 0.50, capa as-of 0.43, LightRAG 0.60. En LongMemEval, sobre los casos evaluados: 0.14, 0.14 y 0.40.

## Piloto de verificación del 2026-11-06 (solo Experimento 2, 6.78 unidades)

- **Resultado:** lanzador con código 0, manifiesto `completado`, análisis y las dos figuras generadas. El Experimento 2 tardó 25.8 minutos y el pipeline completo 30.2.
- **Filas con error:** 6 de 120 (5%), todas de Graphiti en LongMemEval: 3 de 10 casos cortados en 16384 tokens (`08e075c7`, `184da446` y `gpt4_0b2f1d21`), cada uno en sus dos brazos. LightRAG y los escenarios sintéticos no tuvieron errores.
- **Accuracy sobre lo evaluado (solo para depurar):**
  - sintéticos: Graphiti 0.53, capa as-of 0.47, LightRAG 0.53;
  - LongMemEval: 0.29, 0.29 y 0.30.
- **Tiempos:** con 4 grupos a la vez, un lote de 4 escenarios sintéticos tardó 2.2 minutos y uno de 4 casos de LongMemEval unos 9.6.
- **Atención en la corrida final:** si Graphiti falla en el 30% de los casos de LongMemEval, el total de filas con error será de alrededor del 11%, porque allí los escenarios sintéticos pesan menos.

## Cambios tras el piloto (decisión del autor: ajustar la generación)

- **Graphiti:** mantiene su tope por defecto de 16384 tokens. Una salida cortada cuenta como error del caso y no se reintenta (`_NoRetryOnTruncation` en `src/exp2_bitemporal.py`).
  - Un tope de 4096 se probó en el piloto de verificación y se descartó: cortó casos que funcionaban con 16384 (`c9f37c46` y `08f4fc43`).
- **Lanzador:** informa del estado del proceso antes de leer el log en Drive, y registra el motivo de cada consulta sin respuesta. En el piloto de verificación, 5 consultas seguidas sin estado hicieron que la cadena apagara la VM en pleno Experimento 2.
- **Experimento 2:** Graphiti procesa 4 grupos a la vez (`exp2_parallel_groups`) y LightRAG sigue de a uno. Esto no afecta a lo que se mide, porque este experimento no toma tiempos.
- **Wilson:** el intervalo siempre contiene p.
- **Lanzador:** detecta cuando Colab pide reautorizar. `sync_results.py` copia desde Google Drive en el Mac si la VM no responde.
- **`--pilot --skip-exp1`:** el piloto reutiliza el Experimento 1 del piloto anterior, que está en Drive, y ejecuta el Experimento 2, el análisis y las figuras. Solo se permite en el piloto.
- **vLLM fijado en 0.33.0**, la versión de los dos pilotos.
- **Raíz del repo ordenada:**
  - la corrida inválida pasa a `invalidos_20261105/`;
  - el piloto 2, parcial, pasa a `piloto_v2_20261105/`.

## Pasos pendientes

1. ~~Hecho el 2026-11-06.~~ Piloto de verificación: `python3 scripts/colab_launch.py -s pfc --pilot --skip-exp1` (~35 min, ~5 unidades). Apagar la VM con `colab stop` solo si el lanzador termina con código 0; si no, revisar la VM antes. Para pasar, debe cumplir todo esto:
   - menos del 10% de filas con error;
   - el manifiesto termina `completado`;
   - se generan las dos figuras;
   - con esos datos se recalcula la estimación del Experimento 2.
2. Congelar el diseño: el autor hace el commit y `git tag v1-final`. Después no cambiar nada al ver los resultados finales.
3. Corrida final: `python3 scripts/colab_launch.py -s pfc --fresh`, siempre en la misma máquina. Después del 2026-11-13, cuando llegan las 200 unidades.
4. Anotar a mano `muestra_revision_juez.csv`, y escribir método, amenazas a la validez y análisis de errores.
5. Actualizar o retirar `notebooks/` y `docs/COLAB_PRO_CONFIG.md`, que están desactualizados.

## Estimación de la corrida final (con los datos del piloto)

- **Experimento 1, hasta 1M:** ~73 unidades, con 3 corridas más la calidad en 1M. Con las 6 particiones originales eran ~207.
- **Experimento 2:** unas 8 horas y ~71 unidades, medido en el piloto de verificación: 48 lotes de LongMemEval a 9.6 minutos y 8 lotes sintéticos a 2.2.
- **Total:** ~144 unidades. Supone que LightRAG escala de forma lineal; de 100k a 250k subió un 7% en s/MB.
- **Saldo:** 118.64 unidades tras el piloto de verificación, más 200 el 2026-11-13: unas 318, cuya mitad es ~159. La corrida cabe, con unas 15 unidades de margen. Antes del 13 no cabe.

## Condiciones acordadas para una ejecución desatendida

Cortar, apagar la VM y reportar si:

- Algo pide autenticación o una acción manual.
- El smoke test falla, o Graphiti deja de invalidar hechos. Las fechas mal extraídas ya no detienen la corrida: se miden y se reportan.
- No hay A100 disponible tras unos 15 minutos. No sustituirla por otra máquina.
- El piloto muestra más del 10% de filas con error, o un sistema falla de forma sistemática por algo distinto de la extracción de fechas (caídas, errores o respuestas vacías). La baja accuracy de la capa as-of causada por esas fechas no detiene la corrida.
- La estimación de la corrida final supera la mitad del saldo.
- Hace falta cambiar el diseño experimental.

### Corrida final (decisión del autor, 2026-11-06)

Que Graphiti corte salidas en 16384 tokens es un resultado esperado: se reporta y no detiene la corrida. En el piloto de verificación pasó en 3 de 10 casos de LongMemEval, así que en la corrida final se espera alrededor del 11% de filas con error. El límite del 10% solo valía para el piloto.

Cortar, apagar la VM y reportar si:

- Se cae vLLM o Neo4j, o la corrida se detiene por 10 grupos seguidos con fallos.
- Aparecen errores de otro tipo: excepciones distintas del corte en 16384, respuestas vacías o fallos del juez. En cualquier sistema, deben ser más del 2% de las filas.
- Graphiti falla en más del 50% de los casos de LongMemEval, o en algún escenario sintético. En los pilotos no falló en ninguno.
- LightRAG falla en más del 5% de los casos.
- La estimación del tiempo restante supera la mitad del saldo.

Apagar la VM al terminar cada etapa. Los commits del agente son solo locales: el push lo hace el autor.

## Decisiones de diseño

| Tema | Decisión |
|---|---|
| Modelo | `Qwen/Qwen2.5-7B-Instruct` (también es el del ejemplo del repo oficial de E²GraphRAG) |
| Dataset | `xiaowu0162/longmemeval-cleaned`, `longmemeval_oracle.json` (2.83M tokens, 940 sesiones únicas) |
| Máquina final | A100, porque E²GraphRAG usa contextos largos y las mediciones deben ser comparables |
| Base de Graphiti | Neo4j Community 5.26.21 en la VM, solo en localhost |
| E²GraphRAG | Portado del [repo oficial](https://github.com/YiboZhao624/E-2GraphRAG), commit `f598d4f`, licencia MIT |
| Experimento 1 | 100k, 250k, 500k y 1M tokens, 3 corridas, y calidad con 50 preguntas sobre 1M. Se quitaron 1.5M y 2.5M por presupuesto (decisión del autor, 2026-11-05) |
| Experimento 2 | LightRAG, Graphiti sin filtro y Graphiti + capa as-of |
| Casos | 191 preguntas de LongMemEval y 30 escenarios sintéticos (150 preguntas) |
| Piloto | Semilla distinta y 20 preguntas reservadas, que la corrida final no evalúa |
| Errores | Un caso con error se salta y se reintenta al reanudar. La corrida se detiene si cae el servidor o tras 10 grupos seguidos con fallos |
| Estadística | Wilson, McNemar con Holm en comparaciones planificadas, y accuracy contando errores como fallos |

## Qué declarar en el paper

- **Alcance de Graphiti:** los resultados valen para Graphiti con Qwen 7B y configuración por defecto, no para su arquitectura en general. No se puede separar del todo el efecto del modelo del de los prompts de Graphiti.
- **Extracción:** reportar `tasa_fecha_oro_correcta`, `proporcion_hechos_no_latinos` y `fallo_fecha_extraida`. La proporción detecta otros alfabetos, no el idioma, y un valor traducido cuenta como `fallo_extraccion`. Si las fechas fallan de forma general, el experimento muestra que la extracción es el cuello de botella, pero no permite concluir si filtrar con fechas correctas ayuda.
- **Capa as-of:** es propia y actúa sobre los metadatos de Graphiti, que no ofrece consultas as-of. No afirmar que Graphiti las resuelve por sí mismo.
- **Límite de Graphiti:** solo invalida un hecho si el nuevo tiene una fecha de validez estrictamente posterior, así que se espera debilidad en `retroactive-correction`. Reportarlo como hallazgo, sin parchearlo.
- **Alcance:** solo los escenarios sintéticos prueban la bi-temporalidad. LongMemEval evalúa razonamiento temporal general, y allí la capa as-of es exploratoria.
- **Sesgo:** los escenarios los diseñó el autor, el generador es público, y el piloto comparte sujetos y plantillas con la corrida final.
- **Independencia:** las 5 preguntas de cada escenario no son independientes. Reportar también que son 30 escenarios.
- **E²GraphRAG:** los prompts dicen "conversation history" en vez de "novel", se corrigieron dos fallos en sus identificadores de fragmentos, y el límite de 1200 tokens por resumen es una decisión propia.
- **Particiones:** son prefijos del mismo corpus, así que comparten sesiones.
- **Tamaño máximo:** el escalado se mide hasta 1M de los 2.83M tokens del corpus, por el costo de LightRAG en una A100. No extrapolar más allá de 1M.
- **Reintentos de Graphiti:** una salida cortada no se reintenta. Es la única desviación de su configuración por defecto, no toca sus prompts y, con temperatura 0, en el piloto no cambió ningún resultado.
- **Bucles de repetición de Qwen:** reportar los avisos de relaciones descartadas de LightRAG y los casos de Graphiti cortados en 16384 tokens.
- **Tasas de error:** reportarlas por sistema, y no solo la accuracy sobre lo evaluado.
- **Sensibilidad** (filas `sensibilidad` de `resumen_estadistico.csv`): accuracy sobre los casos que todos los sistemas evaluaron, y McNemar planificado contando los errores como fallos.
  - Si las conclusiones cambian respecto del análisis principal, decirlo.
  - En el piloto, LightRAG en LongMemEval bajó de 0.30 a 0.14 en los casos comunes: los casos que Graphiti no completó no son aleatorios.
- **Métricas:** `peak_vram_embeddings_gb` no incluye la memoria del servidor LLM.
