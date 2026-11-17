"""Lectura de la OBT desde Snowflake."""
import os
from pathlib import Path

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

TABLA_OBT = "PSET2_ROSSMANN.OBT.OBT_SALES_DAILY"


def conectar():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        private_key_file=str(RAIZ / "secrets" / "rsa_key.p8"),
        enable_server_session_keep_alive_auto_detection=False,
    )


def cargar_obt() -> pd.DataFrame:
    with conectar() as conn:
        df = conn.cursor().execute(f"SELECT * FROM {TABLA_OBT}").fetch_pandas_all()
    df.columns = df.columns.str.lower()
    return df

def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de ventas con rezago >= 42 días (horizonte máximo del pronóstico)."""
    s = df[["store_id", "sales_date", "sales", "is_model_eligible"]].copy()
    s["sales_date"] = pd.to_datetime(s["sales_date"])
    s["sales_elig"] = s["sales"].where(s["is_model_eligible"])
    tabla = s.pivot(index="sales_date", columns="store_id", values="sales_elig").asfreq("D")

    # Feature 1: media de ventas de los 28 días entre -69 y -42
    f1 = tabla.rolling(28, min_periods=1).mean().shift(42)

    # Feature 2: media de ventas del mismo día de la semana en los días -42, -49, -56 y -63
    f2 = pd.concat([tabla.shift(d) for d in (42, 49, 56, 63)]).groupby(level=0).mean()

    nuevas = pd.concat(
        [f1.stack().rename("sales_mean_lag42_28d"),
         f2.stack().rename("sales_mean_lag42_dow")],
        axis=1,
    ).reset_index()

    df = df.copy()
    df["sales_date"] = pd.to_datetime(df["sales_date"])
    return df.merge(nuevas, on=["store_id", "sales_date"], how="left")