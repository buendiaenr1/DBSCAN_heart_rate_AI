# DBSCAN_heart_rate_AI
Linear modeling of heart rate using DBSCAN of the RUST Language, for the classification of measurements. Taking only heart rate as the dependent variable and age as the independent variable.

### Métodos

El uso el Lenguaje de programación Rust (15) para desarrollar la aplicación de software de esta investigación, tomando como base el paquete (crate es el nombre genérico para los paquetes de dicho lenguaje) denominado petal_clustering (16) para usar DBSCAN como herramienta de clasificación de mediciones multicitada en la literatura científica; considerando que la minería de datos tiene que ver con las técnicas de análisis de datos; es útil para extraer patrones ocultos e interesantes de conjuntos de datos; las técnicas de agrupación son importantes cuando se trata de extraer conocimiento de datos recopilados de diversas aplicaciones, incluidos SIG, imágenes satelitales, cristalografía de rayos X, detección remota y evaluación y planificación ambiental, etc. (17), el algoritmo está descrito en (18), con los parámetros: radio de vecindad de 3.0 y número mínimo de puntos necesarios para formar una región densa de 2; considerando que el error aceptable para los residuales está en un rango de 2 a 11 latidos por minuto (lpm), ver tabla 5 de (19); la ecuación de Tanaka que es la que tiene menos error, pero varía entre 4 y 7 lpm (20) que también se concluye como la mejor para cierto grupo de personas con protocolo de actividad física específica (21).
Las mediciones deben estar en un archivo csv con dos columnas con título fc y edad, y los datos correspondientes a cada columna, este archivo si se crea por ejemplo con Excel se debe guardar como un archivo csv de msdos.
### Análisis estadístico
Se usó regresión lineal con la variable dependiente frecuencia cardiaca (que definida por el usuario y el tipo de mediciones usadas para su análisis puede ser frecuencia cardiaca máxima, frecuencia cardiaca en reposo, entre otras clasificaciones, usos o aplicaciones) y la variable independiente edad, para modelar los grupos de mediciones obtenidos por DBSCAN.
Con respecto a las métricas de evaluación del modelo de regresión se tomaron en cuenta las siguientes:
Para saber que tan preciso es el modelo se usó el error cuadrático medio (MSE) de acuerdo con (22) y como se confirma en (23) MSE=1/n ∑ (actual - estimado)2 y RMSE=raíz (MSE).
La descripción de los residuales mostrados en la aplicación se ejemplifica en (24). Algunas buenas características de los residuales son: alta densidad de puntos cercanos al origen, es simétrica en el origen, tienen distribución normal, se usan para validar el modelo de regresión (25).
Si RSS es la suma de los residuales al cuadrado ∑ (actual - estimado)2. TSS es ∑ (actual – media de las estimaciones)2, así que R2 = (TSS-RSS)/TSS, que da el grado de variabilidad en la variable objetivo que se explica por el modelo o las variables independientes; por otro lado Rajustada=1-((1-R2)(n-1))/(n-k-1) con: n es el número de datos, R2 es la variabilidad en la variable objetivo, k es número de variables independientes (26).
Para el caso de probar los supuestos de los residuales solo se usó: 
Para verificar la normalidad de los residuales con el Test D’agostino – Pearson traducción propia de los autores de esta investigación del lenguaje Matlab a Rust de (27),  el algoritmo se describe en (28). 
Para verificar autocorrelación el estadístico Durbin Watson (DW); el estadístico de Durbin-Watson siempre tendrá un valor entre 0 y 4. Un valor de 2.0 indica que no se detecta autocorrelación en la muestra; los valores de 0 a menos de 2 apuntan a una autocorrelación positiva y los valores de 2 a 4 significan una autocorrelación negativa; una regla general es que los valores estadísticos de la prueba DW en el rango de 1.5 a 2.5 son relativamente normales. Sin embargo, los valores fuera de este rango podrían ser motivo de preocupación, el estadístico de Durbin-Watson, la muestran muchos programas de análisis de regresión (29).


#### Ejemplo de uso: 
En MSDOS en la misma carpeta frec.exe , fc.csv
* frec.exe < fc.csv > resultados.txt
donde: 
* fc.csv tiene dos columnas fc y edad
* resultados.txt es la información generada por la aplicación
