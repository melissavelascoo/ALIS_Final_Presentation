# ALIS — Asistente de Lengua de Señas Boliviana

**ALIS** es una plataforma web que busca facilitar la comunicación entre la Lengua de Señas Boliviana (LSB) y el español mediante el uso de inteligencia artificial y visión por computadora.

El proyecto se encuentra en fase de **prototipo funcional**, con soporte para las cinco vocales del alfabeto dactilológico: **A, E, I, O y U**.

## Funcionalidades

ALIS ofrece dos modalidades de traducción:

- **LSB → Español:** reconocimiento de vocales mediante la cámara en tiempo real, con la posibilidad de construir texto a partir de los caracteres detectados.
- **Español → LSB:** representación visual de las vocales mediante imágenes y reproducción secuencial de las señas disponibles.

Actualmente, el sistema trabaja con el alfabeto dactilológico y no realiza una traducción lingüística completa de la LSB.

## Tecnologías utilizadas

- **Frontend:** HTML5, CSS3 y JavaScript.
- **Backend:** Python 3.11, FastAPI y Uvicorn.
- **Visión por computadora:** OpenCV, MediaPipe y cvzone.
- **Aprendizaje automático:** TensorFlow, Keras y Teachable Machine.
- **Procesamiento de datos:** NumPy.

## Estructura del proyecto

ALIS está organizado en dos módulos principales:

### HandSignDetector

Módulo utilizado durante el desarrollo del modelo de reconocimiento. Permite recopilar imágenes de las señas mediante la cámara y realizar pruebas de clasificación.

Las imágenes recolectadas se utilizaron para entrenar el modelo inicial en **Google Teachable Machine**, desde donde se exportó en formato TensorFlow/Keras.

Contiene los siguientes directorios:

- `Data/`: imágenes recopiladas para el entrenamiento, almacenadas localmente.
- `Model/`: modelo entrenado (`keras_model.h5`) y archivo de etiquetas (`labels.txt`).
- `DCT/`: (DCT = Data Collection & Testing) scripts de recolección de imágenes y pruebas del modelo.

### WebSite

Plataforma web de ALIS que integra las funcionalidades de traducción y utiliza el modelo previamente entrenado.

Está organizada en:

- `frontend/`: interfaz desarrollada con HTML, CSS y JavaScript. Gestiona la cámara, la visualización de resultados y la representación de las señas.
- `backend/`: servidor desarrollado con FastAPI que recibe las imágenes, procesa la detección de manos y ejecuta el modelo de clasificación para devolver las predicciones.

Ambos módulos forman parte del mismo proyecto. `HandSignDetector` contiene los recursos utilizados para desarrollar y probar el modelo inicial, mientras que `WebSite` implementa la plataforma funcional de ALIS.

## Instalación y ejecución

### Requisitos previos

- Python 3.11.
- Cámara web o integrada.
- Archivos `keras_model.h5` y `labels.txt` en `HandSignDetector/Model/`.

### 1. Obtener el proyecto

Clonar el repositorio o descargarlo desde GitHub.

### 2. Configurar el entorno

Desde la carpeta principal de ALIS, ejecutar en Windows PowerShell:

```powershell
cd WebSite/backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

```powershell
python -m pip install -r requirements.txt
```

### 4. Iniciar el servidor

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Abrir la plataforma en:

**http://localhost:8000**

Para utilizar el reconocimiento de señas, permitir el acceso a la cámara desde el navegador.

## Documentación

El informe académico y la documentación complementaria del proyecto se encuentran en la carpeta [docs](./docs/).

## Estado del proyecto

**Prototipo funcional en desarrollo.**

La versión actual permite reconocer y representar las vocales A, E, I, O y U. Se contempla ampliar el reconocimiento al alfabeto dactilológico completo y mejorar el modelo en futuras etapas.