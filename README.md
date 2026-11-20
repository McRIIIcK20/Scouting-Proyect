# Data-Driven Football Scouting: Modelo Multicriterio para Delanteros (Enfoque Moneyball)

## Resumen del Proyecto
Este proyecto aplica técnicas de ciencia de datos e investigación de operaciones para identificar delanteros centro infravalorados en el mercado de la Premier League. Utilizando el algoritmo multicriterio **TOPSIS**, el modelo evalúa métricas ofensivas avanzadas (Goles Esperados, Asistencias Esperadas, Volumen de Tiros) penalizando simultáneamente el valor de mercado del jugador para encontrar eficiencias económicas ("Moneyball").

## Tecnologías y Herramientas
* **Lenguaje:** Python (Pandas, NumPy)
* **Entorno:** Visual Studio Code, Jupyter Notebooks
* **Fuentes de Datos:** FBref (Standard/Shooting), Understat (xG/xA), Transfermarkt (Valoración económica vía Kaggle).
* **Control de Versiones:** Git & GitHub

## Metodología
1. **Ingeniería de Datos:** Extracción y consolidación de 5 temporadas de datos (2020-2026).
2. **Limpieza y Normalización:** Cruce relacional de más de 800 registros mediante estandarización de cadenas de texto y resolución de duplicados (`merge_asof`).
3. **Restricciones de Modelo:** Filtro de minutos mínimos (>900 min) para asegurar significancia estadística.
4. **Modelado Matemático (TOPSIS):** Construcción de una matriz de decisión donde se maximizan atributos de rendimiento (NPxG/90, SoT/90, xA/90) y se minimiza el costo (Valor de Mercado en Euros).

## Hallazgos Principales (Mercado 25/26)
El algoritmo filtró a los perfiles de alto costo mediático, revelando las oportunidades de mercado más eficientes:
* **Beto (Everton):** Lidera el ranking de eficiencia (Score: 0.7633) al ofrecer una producción ofensiva de élite (0.63 NPxG/90) por un valor de mercado contenido de €20M.
* **Callum Wilson (West Ham):** Representa la opción de bajo riesgo financiero más sólida. Con un costo de apenas €5M, mantiene métricas sobresalientes (0.55 NPxG/90).

## Cómo ejecutar este proyecto
1. Clona el repositorio: `git clone https://github.com/tu-usuario/Scouting-project.git`
2. Activa el entorno virtual e instala dependencias: `pip install -r requirements.txt`
3. Ejecuta los cuadernos en la carpeta `/notebooks` en orden secuencial.

## ⚖️ Validación Cruzada (AHP + MABAC)
Para dotar al modelo de rigor científico y eliminar el sesgo humano en la asignación de pesos, se integró el **Proceso de Análisis Jerárquico (AHP)** mediante matrices de Saaty (priorizando NPxG y Costo). 
Posteriormente, el top de jugadores se sometió a una validación cruzada utilizando el algoritmo **MABAC** (Multi-Attributive Border Approximation Area). 

**Resultado:** Perfiles como **Beto** y **Callum Wilson** mantuvieron sus posiciones en el Top 4 en ambos algoritmos matemáticos (TOPSIS y MABAC), confirmando empíricamente su estatus como oportunidades de máxima eficiencia financiera y deportiva.
## 🧮 Laboratorio MCDM: 16 métodos de ranking + 8 esquemas de pesos
La app (`streamlit run app.py`) ahora compara **todos los métodos vistos en el ramo Toma de Decisiones (MCDM)** sobre la misma matriz de decisión (NPxG/90 ↑, Tiros al arco/90 ↑, xA/90 ↑, Valor de mercado ↓).

| Tipo | Métodos | Dónde |
|---|---|---|
| **Pesos subjetivos** | AHP (con CR), SWARA, BWM (con ξ y CR), ANP (supermatriz límite) | `config/settings.yaml` |
| **Pesos objetivos** | Entropía, MEREC, ITARA | se calculan con los jugadores filtrados |
| **Ranking** | TOPSIS, MABAC, MEREC, SAW, COPRAS, VIKOR, MOORA, MULTIMOORA, WASPAS, CoCoSo, MAIRCA, DN-MARCOS, RAFSI, Grey (GRA), Fuzzy TOPSIS, Three-Way | `core/methods.py` |

En la barra lateral se elige el **método de pesos** y el **modelo de ranking**. La sección *Comparación de métodos MCDM* muestra:
* **Consenso:** ranking promedio y rango mejor–peor posición de cada jugador entre los métodos elegidos.
* **Posición por método:** mapa de calor de la posición de cada jugador en cada método.
* **Concordancia:** correlación de Spearman entre los rankings (qué métodos piensan parecido).
* **Three-Way:** shortlist *Aceptar / Frontera / Rechazar* con umbrales α y β ajustables.
* **Comparación de pesos:** los 8 esquemas de ponderación lado a lado.

### Validación
`tests/test_methods.py` reproduce los resultados de las planillas del ramo (CoCoSo, MAIRCA, DN-MARCOS, SWARA, ITARA, MOORA, MULTIMOORA, Entropía, MEREC, SAW, TOPSIS, COPRAS, MABAC y VIKOR) y verifica propiedades básicas (dominancia, monotonía del costo, rankings válidos) de los 16 métodos.

Notas de implementación:
* **MEREC** ahora divide por *n* (no por *n−1*) al quitar un criterio, como en Keshavarz-Ghorabaee et al. (2021) y la guía del ramo, y reemplaza los ceros por el menor valor positivo de la columna (antes un xA = 0 se llevaba casi todo el peso).
* **Fuzzy TOPSIS** fuzzifica cada métrica como número triangular (x·0.9, x, x·1.1) porque los datos son nítidos.
* **RAFSI** usa como ideal/anti-ideal los extremos observados y *n₂ₖ* = N° de jugadores (como en el formulario).
* **DN-MARCOS** usa 1 − x/‖x‖ en la parte vectorial de criterios de costo; con `vector_cost="raw"` replica exactamente la planilla del ramo.
* **ITARA** usa la normalización del formulario (x/max, min/x) y un umbral de indiferencia de 0.01.

Referencias de los métodos que no tenían planilla: Rezaei (2015, 2016) para BWM; Saaty (1996) para ANP; Žižović et al. (2020, *Mathematics* 8(6):1015) para RAFSI; Chen (2000) para Fuzzy TOPSIS; Deng (1989) para el análisis relacional gris; Yao (2010) para Three-Way Decisions; Zavadskas et al. (2012) para WASPAS; Yazdani et al. (2019) para CoCoSo.

```bash
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py
```
