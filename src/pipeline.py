"""Preparación de datos, preprocesamiento, métricas, validación y registro del PSet 3."""
import hashlib
import subprocess
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

INICIO_PRUEBA = pd.Timestamp("2015-06-20")
TARGET = "sales"

NUMERICAS = ["sales_mean_lag42_28d", "sales_mean_lag42_dow", "day_of_month"]
NUM_ESTRUCTURALES = ["weeks_since_promo2_start", "months_since_competition_open"]
DISTANCIA = ["competition_distance_m"]
BOOLEANAS = [
    "is_promo", "is_promo2_active", "participates_promo2", "is_school_holiday",
    "is_state_holiday", "was_open_prev_day", "is_open_next_day", "was_promo_prev_day",
    "is_state_holiday_prev_day", "is_state_holiday_next_day",
]
CATEGORICAS = ["store_type", "assortment", "state_holiday_type", "competition_status",
               "day_of_week_iso", "month"]

FEATURES = NUMERICAS + NUM_ESTRUCTURALES + DISTANCIA + BOOLEANAS + CATEGORICAS


def preparar_datos(df: pd.DataFrame):
    """Deja las filas válidas para el modelo y separa entrenamiento y prueba por fecha."""
    d = df[df["is_model_eligible"]]
    d = d.dropna(subset=["sales_mean_lag42_28d", "sales_mean_lag42_dow"])
    d = d.sort_values(["sales_date", "store_id"]).reset_index(drop=True)
    d[BOOLEANAS] = d[BOOLEANAS].astype(float)

    entrenamiento = d[d["sales_date"] < INICIO_PRUEBA].reset_index(drop=True)
    prueba = d[d["sales_date"] >= INICIO_PRUEBA].reset_index(drop=True)
    return entrenamiento, prueba


def crear_preprocesador() -> ColumnTransformer:
    """Imputación, codificación y escalado. Dentro del Pipeline se ajusta solo con entrenamiento."""
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    estructurales = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
        ("escalar", StandardScaler()),
    ])
    distancia = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("log", FunctionTransformer(np.log1p)),
        ("escalar", StandardScaler()),
    ])
    booleanas = SimpleImputer(strategy="most_frequent")
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numericas, NUMERICAS),
        ("estructurales", estructurales, NUM_ESTRUCTURALES),
        ("distancia", distancia, DISTANCIA),
        ("bool", booleanas, BOOLEANAS),
        ("cat", categoricas, CATEGORICAS),
    ])


K_COSTO = 2.0


def wape(y_real, y_pred) -> float:
    """Error absoluto total dividido entre las ventas totales."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_real - y_pred).sum() / y_real.sum()


def costo_turnos(y_real, y_pred, k: float = K_COSTO) -> float:
    """Costo relativo de turnos mal asignados: pasarse pesa 1, quedarse corto pesa k."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sobre = np.clip(y_pred - y_real, 0, None).sum()
    corto = np.clip(y_real - y_pred, 0, None).sum()
    return (sobre + k * corto) / y_real.sum()


wape_scorer = make_scorer(wape, greater_is_better=False)
costo_scorer = make_scorer(costo_turnos, greater_is_better=False)

CORTES_CV = ["2015-02-14", "2015-03-28", "2015-05-09"]
DIAS_VALIDACION = 42


def pliegues_temporales(entrenamiento: pd.DataFrame) -> list:
    """Pliegues de ventana creciente por fecha: entrena con todo lo anterior y valida 42 días."""
    fechas = entrenamiento["sales_date"]
    pliegues = []
    for inicio in pd.to_datetime(CORTES_CV):
        fin = inicio + pd.Timedelta(days=DIAS_VALIDACION)
        idx_entrena = np.where(fechas < inicio)[0]
        idx_valida = np.where((fechas >= inicio) & (fechas < fin))[0]
        pliegues.append((idx_entrena, idx_valida))
    return pliegues


TRACKING_URI = "http://localhost:5000"
EXPERIMENTO = "rossmann"
TABLA_ORIGEN = "PSET2_ROSSMANN.OBT.OBT_SALES_DAILY"
SEMILLA = 42


def configurar_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)


def commit_git() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def hash_datos(X: pd.DataFrame, y: pd.Series) -> str:
    h = pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def crear_modelo(estimador, preprocesar: bool = True) -> Pipeline:
    if preprocesar:
        return Pipeline([("prep", crear_preprocesador()), ("modelo", estimador)])
    return Pipeline([("modelo", estimador)])


class ReglaNegocio(BaseEstimator, RegressorMixin):
    """Regla del PSet 1: promedio del mismo día de la semana en las 4 semanas previas
    al pronóstico. A 42 días de horizonte son los días -42, -49, -56 y -63."""

    def fit(self, X, y):
        self.respaldo_ = float(np.median(y))
        return self

    def predict(self, X):
        return X["sales_mean_lag42_dow"].fillna(self.respaldo_).to_numpy()


def evaluar_y_registrar(estimador, nombre: str, familia: str, fase: str,
                        entrenamiento: pd.DataFrame, preprocesar: bool = True):
    """Validación cruzada + entrenamiento final + registro completo en MLflow."""
    X = entrenamiento[FEATURES]
    y = entrenamiento[TARGET]
    modelo = crear_modelo(estimador, preprocesar)

    with mlflow.start_run(run_name=nombre) as run:
        cv = cross_validate(
            modelo, X, y, cv=pliegues_temporales(entrenamiento),
            scoring={"wape": wape_scorer, "costo": costo_scorer},
            return_train_score=True,
        )
        inicio = time.time()
        modelo.fit(X, y)
        segundos = time.time() - inicio

        mlflow.log_params({f"modelo__{k}": v for k, v in estimador.get_params().items()})
        mlflow.log_params({
            "semilla": SEMILLA,
            "datos_tabla": TABLA_ORIGEN,
            "datos_fecha_corte": str(entrenamiento["sales_date"].max().date()),
            "datos_obt_construida": str(entrenamiento["obt_built_at"].max()),
            "datos_hash": hash_datos(X, y),
            "datos_filas": len(X),
        })
        mlflow.log_metrics({
            "wape_train": -cv["train_wape"].mean(),
            "wape_cv_media": -cv["test_wape"].mean(),
            "wape_cv_std": cv["test_wape"].std(),
            "costo_cv_media": -cv["test_costo"].mean(),
            "costo_cv_std": cv["test_costo"].std(),
            "segundos_entrenamiento": segundos,
        })
        mlflow.set_tags({"fase": fase, "familia": familia, "git_commit": commit_git()})

        clase = type(estimador)
        ejemplo = X.head(5)
        firma = infer_signature(ejemplo, modelo.predict(ejemplo))
        mlflow.sklearn.log_model(
            modelo, name="modelo", signature=firma, input_example=ejemplo,
            skops_trusted_types=["numpy.dtype", f"{clase.__module__}.{clase.__name__}"],
        )

    return run.info.run_id, cv