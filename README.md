# PSet 3: Experimentación y selección del modelo (Rossmann)

Pronóstico de ventas diarias de 1.115 tiendas Rossmann con 42 días de anticipación, para planificar turnos.
Los datos vienen de la OBT del PSet 2 en Snowflake (`PSET2_ROSSMANN.OBT.OBT_SALES_DAILY`).

## Estructura

```
├── docker-compose.yml      # servidor de MLflow
├── mlflow/Dockerfile       # imagen de MLflow 3.18.0
├── requirements.txt
├── src/
│   ├── datos.py            # conexión a Snowflake, carga de la OBT y features derivadas
│   └── pipeline.py         # preprocesamiento, métricas, validación temporal y registro en MLflow
├── notebooks/              # se ejecutan en orden, de 01 a 06
└── docs/
    ├── corridas.csv        # exportación de todas las corridas de MLflow
    ├── llm_prompts.txt     # prompts de la prueba con LLM
    └── llm_respuestas.jsonl
```

## 1. Entorno de Python

Requiere Python 3.12.

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En VS Code, cada notebook debe usar el kernel `.venv`.

## 2. Configuración de Snowflake

Las credenciales no están en el repositorio. Se necesitan dos archivos en la raíz del proyecto:

- `.env`: copiar `.env.example` como `.env` y completar los valores (cuenta, usuario de servicio, rol, warehouse, base de datos).
- `secrets/rsa_key.p8`: llave privada del usuario de servicio `PSET2_SVC`, la misma del PSet 2.

Ambos están en `.gitignore`.

## 3. Levantar MLflow

Con Docker Desktop abierto:

```
docker compose up -d --build
```

La interfaz queda en http://localhost:5000. El backend es SQLite y los artefactos se guardan en el volumen `mlflow-data`, así que las corridas se conservan al reiniciar el contenedor. Para detenerlo: `docker compose down`.

En Windows con 12 GB de RAM conviene limitar la memoria de WSL creando `C:\Users\<usuario>\.wslconfig` con:

```
[wsl2]
memory=3GB
processors=2
```

## 4. Orden de los notebooks

| Notebook | Contenido |
|---|---|
| `01_eda.ipynb` | Exploración de la OBT y hallazgos |
| `02_features.ipynb` | Features con rezago de 42 días, prueba de ausencia de fuga y prueba del pipeline |
| `03_baseline_candidatos.ipynb` | Dos líneas base, cinco familias de modelos, Chronos-2, prompting con LLM y prueba con log(ventas) |
| `04_tuning.ipynb` | Búsqueda aleatoria de 30 configuraciones para los tres mejores (corridas anidadas) |
| `05_seleccion.ipynb` | Evaluación en prueba, importancia por permutación y registro de champion y challenger en `rossmann-modelo` |
| `06_reproduccion.ipynb` | Carga `models:/rossmann-modelo@champion`, reproduce la métrica y exporta `docs/corridas.csv` |

Los notebooks 03 y 04 tardan varias horas. MLflow debe estar corriendo antes de ejecutar los notebooks 03 a 06.

## Resultado

Champion: HistGradientBoosting (`rossmann-modelo` versión 1). Challenger: Random Forest (versión 2).
En el periodo de prueba (2015-06-20 a 2015-07-31), el champion logra WAPE de 0,090 frente a 0,180 de la regla del gerente, y reduce el costo de turnos mal asignados en 47%.
