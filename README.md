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

#### Referencias

15. 	Rust Language. Install Rust - Rust Programming Language, rustc 1.60.0 (7737e0b5c 2022-04-04) [Internet]. 2022 [cited 2022 May 24]. Available from: https://www.rust-lang.org/tools/install
16. 	Rust Language. petal_clustering - Rust [Internet]. Version 0.5.1. 2022 [cited 2022 May 24]. Available from: https://docs.rs/petal-clustering/latest/petal_clustering/
17. 	Khan K, Rehman SU, Aziz K, Fong S, Sarasvady S, Vishwa A. DBSCAN: Past, present and future. 5th International Conference on the Applications of Digital Information and Web Technologies, ICADIWT 2014. 2014;232–8. 
18. 	Schubert E, Sander J, Ester M, Kriegel HP, Xu X. DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN. ACM Trans Database Syst [Internet]. 2017 Jul 1 [cited 2022 May 27];42(3):19:1-19:21. Available from: https://api.semanticscholar.org/CorpusID:5156876#id-name=S2CID
19. 	Robergs R, Landwehr R. La sorprendente Historia de la Ecuación (FC máx. = 220 – edad). G-SE [Internet]. 2003 Feb 2 [cited 2022 May 27];0. Available from: https://g-se.com/la-sorprendente-historia-de-la-ecuacion-fc-max.-220-edad-67-sa-457cfb270ee0c9
20. 	No-Ko. FRECUENCIA CARDÍACA MÁXIMA: ¿QUÉ ES? ¿COMO CALCULARLA? | FADE [Internet]. FADE saludable. 2022 [cited 2022 May 27]. Available from: http://fadesaludable.es/2017/02/23/frecuencia-cardiaca-maxima-que-es-como-calcularla/
21. 	Miragaya MA, Magri OF. Ecuación más conveniente para predecir FCM esperada en esfuerzo Ecuación más conveniente para predecir frecuencia cardíaca máxima esperada en esfuerzo. Insuf Card [Internet]. 2016 [cited 2022 May 27];11(2):56–61. Available from: http://www.insuficienciacardiaca.org
22. 	Engelmann P. Simple Linear Regression from Scratch in Rust - CodeProject [Internet]. 2018 [cited 2022 May 26]. Available from: https://www.codeproject.com/Articles/1271862/Simple-Linear-Regression-from-Scratch-in-Rust
23. 	Statistics How To. Mean Squared Error: Definition and Example - Statistics How To [Internet]. 2022 [cited 2022 May 26]. Available from: https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/mean-squared-error/
24. 	NZ Maths. Residual (in linear regression) | NZ Maths [Internet]. New Zealand Government. 2022 [cited 2022 May 26]. Available from: https://nzmaths.co.nz/category/glossary/residual-linear-regression
25. 	Gohar U. How to use Residual Plots for regression model validation? | by Usman Gohar | Towards Data Science [Internet]. Using residual plots to validate your regression models. 2020 [cited 2022 May 26]. Available from: https://towardsdatascience.com/how-to-use-residual-plots-for-regression-model-validation-c3c70e8ab378
26. 	Bhandari A. Difference Between R-Squared and Adjusted R-Squared [Internet]. 2020 [cited 2022 May 26]. Available from: https://www.analyticsvidhya.com/blog/2020/07/difference-between-r-squared-and-adjusted-r-squared/
27. 	Trujillo-Ortiz A, Hernandez-Walls R. D’Agostino-Pearson’s K2 test for assessing normality of data using skewness and kurtosis. [Internet]. A MATLAB file. Facultad de Ciencias Marinas, Universidad Autónoma de Baja California, Apdo. Postal 453, Ensenada, Baja California. 2003 [cited 2022 May 28]. Available from: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/46548/versions/2/previews/Codes_data_publish/Codes/Dagostest.m/index.html
28. 	D’Agostino’s K-squared test - Wikipedia [Internet]. 2022 [cited 2022 May 28]. Available from: https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test
29. 	Kenton W. Durbin Watson Statistic Definition [Internet]. CORPORATE FINANCE & ACCOUNTING  FINANCIAL ANALYSIS. 2021 [cited 2022 May 28]. Available from: https://www.investopedia.com/terms/d/durbin-watson-statistic.asp

