# Proyecto final de Deep Learning aplicado a series temporales meteorológicas


**Autores:** Juan Andrés Ramos Cardona y Sergio Andrés Cadavid Bruges

## Resumen 

Este Jupyter Book presenta el desarrollo completo de un sistema de
pronóstico meteorológico horario basado en Deep Learning, construido sobre
datos históricos del **Instituto Nacional de Meteorología de Brasil
(INMET)**. El proyecto aborda un problema real de forecasting multivariable:
predecir la temperatura del aire a partir de series horarias y covariables
meteorológicas, evaluando de forma rigurosa varias arquitecturas modernas de
aprendizaje profundo.

El trabajo no se limita a entrenar modelos. La propuesta integra definición
formal del problema, análisis exploratorio, ingeniería de datos, diseño
experimental, comparación estadística, discusión de resultados y criterios de
reproducibilidad. El objetivo es ofrecer una evaluación clara, trazable y
defendible ante un jurado académico.

## Motivación

El pronóstico meteorológico de corto plazo es una tarea crítica para sectores
como energía, agricultura, transporte, gestión del riesgo, salud pública y
operación de infraestructura. Aunque los modelos numéricos tradicionales
siguen siendo fundamentales, los enfoques basados en datos han ganado
relevancia por su capacidad para capturar patrones locales, relaciones no
lineales y dependencias temporales complejas.

Las estaciones automáticas del INMET generan observaciones horarias que
permiten estudiar el comportamiento del clima en distintas regiones de Brasil.
Este escenario es especialmente interesante para Deep Learning porque combina
estacionalidad diaria, dinámica semanal, variabilidad regional, valores
faltantes, covariables exógenas y horizontes de predicción multistep.

## Pregunta de investigación

La pregunta central del proyecto es:

> ¿Qué arquitectura de Deep Learning ofrece el mejor desempeño, estabilidad y
> capacidad de generalización para forecasting meteorológico horario sobre un
> panel de estaciones INMET?

Para responderla, el proyecto compara modelos recurrentes, modelos basados en
bloques de forecasting, arquitecturas con atención y una línea base de
persistencia. La evaluación considera no solo el error promedio, sino también
la variabilidad entre semillas, el comportamiento por horizonte, la
generalización por región y la significancia estadística de las diferencias.

## Objetivos

El objetivo general es construir y evaluar un pipeline reproducible para
forecasting meteorológico horario con Deep Learning.

Los objetivos específicos son:

- Formular el problema de forecasting multistep a partir de datos horarios del
  INMET.
- Diseñar un proceso de limpieza, transformación y partición temporal que
  evite data leakage.
- Analizar patrones temporales, distribución de variables, faltantes y
  diferencias regionales.
- Implementar y entrenar modelos comparables bajo una misma configuración
  experimental.
- Evaluar el desempeño con métricas estándar de regresión y análisis por
  horizonte.
- Aplicar pruebas estadísticas para sustentar si las diferencias entre modelos
  son significativas.
- Documentar el flujo de trabajo con criterios de ciencia abierta y
  reproducibilidad.

## Alcance técnico

El estudio trabaja con series horarias de estaciones meteorológicas del INMET.
La variable principal de predicción es la temperatura del aire (`temp_c`) y se
incluyen covariables meteorológicas como humedad, presión, viento,
radiación y punto de rocío, según disponibilidad en los datos procesados.

La tarea experimental se plantea con una ventana histórica de **168 horas**
(siete días) para producir un pronóstico multistep de **24 horas**. Los datos
se dividen cronológicamente para respetar la naturaleza temporal del problema:
entrenamiento, validación y prueba se separan por periodos, evitando mezclar
información futura durante el ajuste de modelos o escaladores.

## Modelos evaluados

El benchmark considera una línea base y cinco familias de modelos:

- **Persistencia:** referencia simple y exigente para forecasting de corto
  plazo.
- **LSTM:** red recurrente capaz de capturar dependencias temporales mediante
  memoria interna.
- **GRU:** alternativa recurrente más compacta, con menor complejidad
  paramétrica.
- **N-BEATSx:** arquitectura especializada en forecasting, extendida con
  variables exógenas.
- **Temporal Fusion Transformer (TFT):** modelo con mecanismos de atención,
  selección de variables e interpretabilidad, usado como arquitectura guía del
  estudio.
- **Informer:** variante eficiente basada en atención, orientada a secuencias
  largas.

Esta selección permite contrastar enfoques clásicos de secuencia, modelos
especializados en forecasting y arquitecturas modernas basadas en atención.

## Metodología de evaluación

La comparación se realiza con métricas de regresión como RMSE, MAE, R² y
sMAPE. Además, los resultados se analizan por modelo, horizonte de pronóstico
y región geográfica.

Para fortalecer la validez de las conclusiones, el benchmark final incluye:

- intervalos de confianza por bootstrap,
- comparación de desempeño entre semillas,
- ranking por horizonte,
- pruebas no paramétricas como Friedman, Nemenyi y Wilcoxon,
- prueba Diebold-Mariano para errores pareados,
- diagnóstico residual con Ljung-Box y BDS,
- análisis del trade-off entre precisión y costo computacional.

Esta metodología busca diferenciar mejoras reales de variaciones atribuibles
al azar, a una semilla favorable o a una región específica.

## Contribuciones del proyecto

Las principales contribuciones son:

- Un pipeline completo para forecasting meteorológico horario con datos reales
  del INMET.
- Un proceso reproducible de preprocesamiento, escalado, partición temporal y
  evaluación.
- Una comparación homogénea entre arquitecturas profundas bajo un mismo marco
  experimental.
- Un análisis exploratorio orientado a entender la estructura temporal y
  regional del problema.
- Un benchmark final con soporte estadístico para sustentar las conclusiones.
- Una documentación en formato Jupyter Book pensada para revisión académica,
  trazabilidad técnica y comunicación clara de resultados.

## Estructura del libro

El libro está organizado en siete capítulos:

1. **Problema y dataset:** definición de la tarea, fuente de datos, variables
   y alcance experimental.
2. **Análisis exploratorio:** estudio de distribución, faltantes, patrones
   temporales y comportamiento regional.
3. **Estado del arte:** revisión de enfoques relevantes en forecasting con
   Deep Learning.
4. **Implementación y entrenamiento:** descripción del pipeline, modelos,
   configuración y artefactos generados.
5. **Paper guía:** análisis de la arquitectura de referencia y su relación con
   el problema.
6. **Benchmark final y tests estadísticos:** comparación cuantitativa,
   significancia estadística, diagnóstico residual y discusión de resultados.
7. **Reproducibilidad:** instrucciones, estructura del repositorio,
   configuración y criterios para regenerar los experimentos.




