# ALIS

**ALIS** es un prototipo para explorar la comunicación entre el español escrito y la Lengua de Señas Boliviana (LSB). El proyecto reúne el desarrollo y evaluación de modelos de reconocimiento en `slr/` y una aplicación web con API en `web/`.

> La versión web reconoce un conjunto de letras y señas entrenadas y puede mostrar el alfabeto dactilológico. La salida Español → LSB es un deletreo visual de apoyo; no traduce frases a la gramática completa de la LSB.

## Funcionalidades actuales

- **LSB → español:** reconocimiento desde la cámara, con selección automática de modelos o modo MLP/LSTM.
  - El **MLP estático** clasifica la forma de una mano a partir de 63 características y suaviza las predicciones de hasta cinco fotogramas.
  - El **LSTM dinámico** procesa secuencias de 30 fotogramas con 454 características por fotograma.
  - En modo automático, el arbitraje prioriza la actividad dinámica y su reconocimiento para evitar que ambos modelos agreguen resultados superpuestos.
  - La interfaz encola los resultados con intervalos entre letras y señas. Incluye la regla contextual para mostrar **ES** cuando se reconoce **SER** después de “HOLA MI NOMBRE”.
- **Español → LSB:** muestra las ilustraciones PNG transparentes del alfabeto y permite recorrerlas o reproducirlas. Los espacios se representan con un icono de pausa y una espera de 2,5 segundos.
- **Interfaz:** diseño adaptable a móvil y escritorio, con transiciones y soporte para la preferencia del sistema de movimiento reducido.
- **API:** expone estado del servicio y modelos, predicción y reinicio del reconocimiento.

## Estructura del proyecto

```text
ALIS/
├── docs/                         # Informe y presentación del proyecto
├── slr/
│   ├── data/
│   │   ├── static_signs/         # Datos de imágenes y landmarks del MLP
│   │   └── dynamic_signs/        # Videos, características y particiones del LSTM
│   ├── models/
│   │   ├── static_signs/         # model.h5 y labels.txt del MLP
│   │   └── dynamic_signs/        # model.h5 y labels.txt del LSTM
│   ├── pipelines/
│   │   ├── static_signs/         # Captura, preparación, entrenamiento y evaluación MLP
│   │   └── dynamic_signs/        # Grabación, extracción, entrenamiento y evaluación LSTM
│   └── reports/                  # Métricas y gráficas generadas por los pipelines
├── web/
│   ├── backend/
│   │   ├── app/api/              # Rutas HTTP
│   │   ├── app/core/             # Configuración
│   │   ├── app/services/         # Extracción de landmarks y clasificación
│   │   ├── requirements.txt      # Dependencias del backend aislado
│   │   └── tests/                # Pruebas del backend
│   └── frontend/
│       ├── assets/css/           # Estilos y animaciones
│       ├── assets/js/            # Cámara, traducción y reproducción
│       └── assets/img/           # Ilustraciones del alfabeto
├── paths.py                      # Rutas compartidas por los pipelines SLR
├── requirements.txt              # Dependencias para el proyecto completo
└── pyproject.toml                 # Metadatos y configuración del paquete
```

Los datasets de entrenamiento y evaluación se guardan bajo `slr/data/`; los modelos y reportes se guardan en `slr/models/` y `slr/reports/`. El backend busca los modelos en esas rutas mediante `paths.py`.

## Requisitos

- Windows, macOS o Linux.
- Python **3.11 recomendado** para trabajar con las versiones actuales de TensorFlow y MediaPipe.
- Cámara web para captura, predicción en vivo y uso de la traducción LSB → español.
- Permiso de cámara en el navegador; `localhost` y HTTPS son contextos seguros para acceder a ella.

## Instalación

Desde la raíz del repositorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` instala las dependencias de los pipelines SLR y del servidor web. `web/backend/requirements.txt` contiene solo las dependencias del backend; elige uno de los dos archivos según el alcance de tu entorno, no es necesario instalar ambos.

## Ejecutar la aplicación web

Con el entorno virtual activado, desde la raíz:

```powershell
cd web\backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Abre <http://127.0.0.1:8000/>. La documentación interactiva de la API está en <http://127.0.0.1:8000/docs>.

Rutas de la API:

| Método | Ruta | Uso |
|---|---|---|
| `GET` | `/api/health` | Comprueba que la API responde. |
| `GET` | `/api/model-status` | Informa qué modelos están cargados y sus clases. |
| `POST` | `/api/predict` | Recibe un fotograma y el modo de reconocimiento. |
| `POST` | `/api/reset` | Reinicia el estado temporal del reconocedor dinámico. |

`/api/predict` acepta los modos `both`, `auto`, `static` y `dynamic`; la interfaz ofrece selección automática, solo MLP o solo LSTM. La petición incluye el fotograma y el modo como datos de formulario.

Umbrales configurables mediante variables de entorno (valores predeterminados):

| Variable | Predeterminado | Aplicación |
|---|---:|---|
| `ALIS_CONFIDENCE_THRESHOLD` | `0.75` | Confianza mínima del modelo dinámico. |
| `ALIS_MLP_CONFIDENCE_THRESHOLD` | `0.70` | Confianza mínima del modelo estático. |

En PowerShell, define una variable antes de iniciar Uvicorn, por ejemplo:

```powershell
$env:ALIS_CONFIDENCE_THRESHOLD = "0.80"
```

## Pipelines SLR

Los comandos se ejecutan desde la raíz con el entorno virtual activado. La captura requiere una cámara; los pasos posteriores requieren los datos generados por los pasos anteriores.

### MLP: señas estáticas

1. `python -m slr.pipelines.static_signs.capture_images` — captura imágenes y landmarks por letra y sesión.
2. `python -m slr.pipelines.static_signs.preprocess_mlp` — limpia y normaliza los 63 valores de landmarks.
3. `python -m slr.pipelines.static_signs.splits_mlp` — crea particiones de entrenamiento, validación y prueba.
4. `python -m slr.pipelines.static_signs.train_mlp` — entrena el clasificador y guarda el modelo, las etiquetas y los reportes.
5. `python -m slr.pipelines.static_signs.test_mlp` — evalúa el conjunto de prueba.
6. `python -m slr.pipelines.static_signs.predict_mlp` — ejecuta la predicción independiente con cámara.

### LSTM: señas dinámicas

1. `python -m slr.pipelines.dynamic_signs.record_videos` — graba secuencias por seña y sesión.
2. `python -m slr.pipelines.dynamic_signs.extract_videos` — extrae secuencias de 30 fotogramas y 454 características por fotograma.
3. `python -m slr.pipelines.dynamic_signs.splits_lstm` — prepara las particiones del dataset.
4. `python -m slr.pipelines.dynamic_signs.train_lstm` — entrena y guarda el modelo, las etiquetas y los reportes.
5. `python -m slr.pipelines.dynamic_signs.test_lstm` — evalúa las particiones de prueba y genera métricas.
6. `python -m slr.pipelines.dynamic_signs.predict_lstm` — ejecuta la predicción independiente en tiempo real.

El entrenamiento puede reemplazar los archivos `model.h5` y `labels.txt` existentes en `slr/models/`. Conserva una copia si quieres mantener los modelos actuales.

## Pruebas

Las pruebas del backend están en `web/backend/tests/`. Si `pytest` aún no está instalado en el entorno:

```powershell
python -m pip install pytest
```

Después:

```powershell
cd web\backend
python -m pytest tests -q
```

## Documentación del proyecto

- [Paper ALIS Final Presentation (PDF)](./docs/Paper%20ALIS%20Final%20Presentation.pdf)
- [Versión editable del documento (DOCX)](./docs/Paper%20ALIS%20Final%20Presentation.docx)
- [Presentación de ALIS](./docs/ALIS_Presentation.pptx)

**Estado:** prototipo funcional en desarrollo. La precisión y cobertura dependen de las clases y los datos usados para entrenar los modelos disponibles.
