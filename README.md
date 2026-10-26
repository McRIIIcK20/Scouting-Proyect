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