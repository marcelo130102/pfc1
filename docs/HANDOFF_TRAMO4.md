# Contexto para continuar: tramo 4 y cierre (actualizado el 2026-11-15)

Proyecto: PFC que compara GraphRAG con un LLM local (Qwen/Qwen2.5-7B-Instruct con vLLM 0.33.0, embeddings BAAI/bge-m3) en Colab A100 High-RAM, controlado desde el Mac con la CLI `colab`.
- **Experimento 1:** eficiencia de indexación, E²GraphRAG vs LightRAG.
- **Experimento 2:** consistencia bi-temporal, Graphiti sin filtro y Graphiti + capa as-of vs LightRAG, con un juez LLM.

El registro detallado de pilotos y tramos está en `docs/ESTADO_EXPERIMENTOS.md`.

## Reglas del autor (obligatorias)

- Responder en español.
- **El autor hace los commits y el push.** No hacer commits.
- **Diseño congelado** en el tag `v1-final`: no tocar código ni configuración, aunque los resultados no favorezcan la hipótesis.
- Usar siempre la misma máquina: A100 High-RAM. Apagar la VM al terminar cada tramo.
- **Parar, apagar la VM e informar si:**
  - hace falta autorización o una acción manual;
  - no hay A100 tras unos 15 minutos (no usar otra máquina);
  - la estimación supera la mitad del saldo;
  - habría que cambiar el diseño.
- Aplicar la regla de corte de la corrida final (está en `ESTADO_EXPERIMENTOS.md`):
  - parar si vLLM o Neo4j caen, o tras 10 grupos seguidos con fallos;
  - parar si otros tipos de error superan el 2%;
  - parar si Graphiti falla en más del 50% de LongMemEval o en cualquier caso sintético;
  - parar si LightRAG falla en más del 5%.

## Estado al 2026-11-15

- **Saldo:** 217.74 unidades, sin VM activa. La renovación se sumó al saldo anterior (17.74 + 200).
- **Experimento 1:** completo, 24 de 24 filas y 50 preguntas de calidad por sistema en 1M.
- **Experimento 2, escenarios sintéticos:** completos, 30 escenarios × 5 categorías × 3 métodos = 450 filas, todas evaluadas.
- **Experimento 2, LongMemEval:** 46 de 191 casos.
  - Graphiti falló en 13 de 46 (28%), en sus dos brazos; todos por el corte de 16384 tokens, aceptado como hallazgo.
  - LightRAG, sin errores.
- **Manifiesto en Drive:** `en_curso`, como corresponde.
- **Respaldos (gitignored):** `final_tramo1_20261106/`, `final_tramo2_20261108/` y `final_tramo3_20261108/`.
- **Resultados vivos:** en Drive, en `~/Library/CloudStorage/GoogleDrive-*/*/PFC_RAG_Resultados/`.

## Tramo 4: qué falta y cómo lanzarlo

**Qué falta:** 145 casos de LongMemEval, entre 6 y 9 horas y entre 55 y 80 unidades. La mitad del saldo es ~109, así que cabe en un tramo.

1. **El autor**, en su terminal: `colab new -s pfc --gpu A100 --high-mem && colab drivemount -s pfc`, autoriza Drive y avisa con "listo". `drivemount` es interactivo y no funciona desde un agente.
2. **Comprobar** con `colab sessions` que hay una sesión `pfc` A100 High-RAM y que no hay un lanzador vivo (`pgrep -fl colab_launch.py`).
3. **Comprobar Drive:**
   ```sh
   (colab exec -s pfc -f scripts/local/drive_check.py 2>&1 & p=$!; sleep 90; kill $p) | tail -5
   ```
   Debe imprimir `DRIVE_OK` y listar `experimento2_resultados.csv`.
4. **Lanzar sin `--fresh`** (reanuda desde los checkpoints), como Bash con `run_in_background`:
   ```sh
   python3 scripts/colab_launch.py -s pfc > <scratchpad>/final_tramo4_launch.log 2>&1
   ```
5. **Lanzar la guardia** en paralelo, también con `run_in_background`:
   ```sh
   SOFT=109 HARD=107 zsh scripts/local/guardia.sh > <scratchpad>/guardia_tramo4.log 2>&1
   ```
   La guardia consulta el saldo y los CSV de Drive cada minuto e imprime `CHECKPOINT …`. Bajo SOFT apaga la VM tras el siguiente checkpoint, y bajo HARD la apaga de inmediato. Con la estimación de 55–80 unidades, la corrida debería terminar antes de que salte.
6. **Seguimiento:**
   - leer `guardia_tramo4.log` y `tail` del log del lanzador;
   - contar filas y errores de LongMemEval con el módulo `csv`;
   - aplicar la regla de corte.

## Al terminar el pipeline

- `python3 scripts/sync_results.py` (si la VM no responde, copia desde Drive).
- Verificar que `run_manifest.json` diga `completado` y que los `code_sha256` coincidan con `v1-final`.
- Comprobar que se generaron `resumen_estadistico.csv`, `muestra_revision_juez.csv` y las figuras.
- `colab stop -s pfc`. Matar el proceso local del lanzador y `caffeinate` si siguen vivos.
- Copiar todo a `final_tramo4_<fecha>/` y a `results/`, que aún no existe.
- Añadir un registro del tramo 4 en `ESTADO_EXPERIMENTOS.md`, con fechas, unidades, casos y errores.

## Trampas operativas conocidas

- `colab exec` no acepta código en línea: usar `-f archivo.py`. Por defecto tiene un tiempo límite de 30 s (`--timeout`).
- En macOS no hay `timeout`: usar `(cmd & p=$!; sleep N; kill $p)`. Los `sleep` encadenados están bloqueados.
- El `python3` del sistema en el Mac no tiene pandas: usar el módulo `csv`.
- Monitor expira a los 30 minutos y mata lo que corre dentro. Por eso la guardia y el lanzador van como Bash con `run_in_background`, y Monitor solo sigue sus logs.
- Sin `--fresh`, el lanzador no relanza si el proceso ya corre en la VM, solo retoma el seguimiento.
- "Allocation refused (precondition failed)" significa que ya existe una sesión: usar esa.
- Hay disponible una actualización de la CLI `colab` (0.8.0, la actual es 0.7.5). No aplicarla a mitad de la corrida.
- Cuando la guardia apaga la VM, el lanzador local termina con error porque se mata a propósito. Es lo esperado.
- El scratchpad de cada sesión es temporal. Los scripts permanentes están en `scripts/local/` (gitignored).

## Estado del análisis (preliminar; lo definitivo sale de `run_analysis` al final)

**Experimento 1:**
- Latencia de indexación, en s/MB: E²GraphRAG unos 62–79 y LightRAG unos 900–1050, es decir ~14× más rápido.
- Calidad en 1M: LightRAG 0.24 (IC 0.14–0.37) y E²GraphRAG 0.02 (0.00–0.10). E²GraphRAG responde con fluidez pero sobre otro tema, lo que apunta a un fallo de recuperación: falta una sección de análisis de errores.

**Experimento 2, escenarios sintéticos:**
- Accuracy: LightRAG 0.53, Graphiti 0.43, capa as-of 0.49.
- McNemar exacto sin corregir: Graphiti vs LightRAG p = 0.09; as-of vs LightRAG p = 0.56. No hay diferencia global significativa, así que la hipótesis "Graphiti rinde mejor" no se confirma en el total.
- Por categoría (exploratorio, no planificado):
  - tt-after-late-arrival: Graphiti gana 10–0 (p = 0.002).
  - retroactive-correction: LightRAG gana 17–2 (p < 0.001).
  - pre-change-audit: as-of 0.90 frente a 0.70 de LightRAG.
  - current-state: LightRAG 0.73 frente a ~0.50.

**¿Implementación o modelo?** Las columnas de diagnóstico indican que falla sobre todo el modelo:
- **Fecha bien extraída:** Qwen pone bien el `valid_at` del hecho correcto en solo 6 de 30 casos en current-state y en retroactive-correction, frente a 25 de 30 en pre-change-audit y tt-before. Suele usar la fecha del mensaje en lugar de la que menciona el texto.
- **Idioma:** el 56% de los hechos del grafo están en chino o ruso, una deriva de idioma de Qwen 7B.
- **Recuperación (implementación):** correcta. Sin filtro, el dato correcto se recupera en 117 de 117 casos en que está en el grafo.
- **Respuesta (modelo):** con hechos contradictorios en el contexto acierta poco. Cuando la capa as-of limpia el contexto, acierta 27 de 30 en pre-change-audit.
- **La capa as-of** funciona con fechas correctas, pero convierte cada error de fecha en un fallo seguro. Es una limitación a declarar, no un bug.

**E²GraphRAG no sirve como alternativa al Experimento 2:**
- no maneja tiempo;
- su calidad es 0.02;
- elegirlo después de ver los resultados sería HARKing.

Hilo conductor posible para la discusión: con LLM locales pequeños, la construcción del grafo es el cuello de botella. Si se delega al LLM (Graphiti) es lenta y propensa a errores; si se usa NLP clásico (E²GraphRAG) es rápida pero pierde calidad. Trabajo futuro: extracción de fechas determinista con el LLM solo para los hechos, o una submuestra con un modelo mayor, separada de la corrida congelada.

## Cierre pendiente (sin cómputo)

- El autor anota a mano `muestra_revision_juez.csv`.
- Redactar:
  - el método;
  - las amenazas a la validez (lista "Qué declarar" en `ESTADO_EXPERIMENTOS.md`; además, que el corte en 16384 tokens no es determinista con temperatura 0, y que los tramos corrieron en distintas VM A100);
  - el análisis de errores (E²GraphRAG 0.02, fechas, idioma);
  - la sensibilidad.
- Actualizar o retirar `notebooks/` y `docs/COLAB_PRO_CONFIG.md`.
- Recordatorio para el autor: `sudo pmset -a disablesleep 0`.
