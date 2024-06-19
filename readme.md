### Prácticas PIVA
##### Curso 2023/2024

Este repositorio contiene el trabajo final de la asignatura _Procesamiento de Imagen, Vídeo y Audio_ del grado en _Ciencia e Ingeniería de Datos_ de la _Universidad de Coruña_.

---
#### Segmentación de carreteras

Se entrena un modelo XGBoost con imágenes satélite obtenidas del [_Massachusetts Roads Dataset_](https://www.cs.toronto.edu/~vmnih/data/) con el objetivo de segmentar las carreteras y obtener una salida binaria. 

Se exploran diferentes aproximaciones, modificando el preprocesado de las imágenes de entrada o el postprocesado de la salida del modelo, utilizando varios espacios de color, filtros y operadores morfológicos.

* main_roads.ipynb: aproximaciones
* utils_roads.py: funciones útiles
* otros/main_roads.html: código ejecutado

---
#### Reconocimiento de animales

Se entrena un modelo XGBoost con imágenes de animales obtenidas del [_Caltech-101 Dataset_](https://data.caltech.edu/records/mzrjq-6wc02) para obtener un clasificador de animales fiable. 

Se exploran diferentes aproximaciones, modificando el preprocesado de las imágenes de entrada, utilizando varios espacios de color, filtros y operadores morfológicos.

* main_animals.ipynb: aproximaciones
* utils_animals.py: funciones útiles
* otros/main_animals.html: código ejecutado