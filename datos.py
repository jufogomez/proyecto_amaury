"""
Datos históricos simulados del banco.

Cada cliente tiene:
    edad            años cumplidos
    ingresos        ingresos mensuales, en millones de pesos
    cuotas          lo que ya paga cada mes por otras deudas, en millones de pesos
    endeudamiento   qué porcentaje de sus ingresos ya se le va en cuotas
    aprobado        1 si el banco le aprobó el crédito, 0 si se lo negó

Los datos se generan con la política del banco, que las IAs NO conocen y
tienen que descubrir por su cuenta a partir de los ejemplos:
    - el endeudamiento no puede pasar del 40%,
    - hay que ganar al menos 1.5 millones al mes,
    - hay que tener entre 21 y 65 años.
Además, un 5% de las decisiones se voltea al azar para imitar las
excepciones de la vida real (casos especiales, errores humanos, etc.).
"""

import numpy as np
import pandas as pd

# Variables con las que razonan los modelos. El endeudamiento no lo escribe
# el usuario: se calcula a partir de ingresos y cuotas ("ingeniería de
# características"), porque así es como piensan los analistas de crédito.
VARIABLES = ["edad", "ingresos", "endeudamiento"]


def calcular_endeudamiento(ingresos, cuotas):
    """Porcentaje de los ingresos que ya se va en cuotas."""
    return np.round(100 * np.asarray(cuotas) / np.asarray(ingresos), 1)


def generar_clientes(n=400, semilla=42):
    """Crea n clientes históricos. La semilla hace que siempre salgan los mismos."""
    rng = np.random.default_rng(semilla)

    edad = rng.integers(18, 71, n)
    # La mayoría gana entre 1.5 y 8 millones; unos pocos ganan mucho más
    ingresos = np.round(np.clip(rng.lognormal(np.log(3.5), 0.55, n), 1.0, 15.0), 1)
    # Qué parte de sus ingresos ya se le va en cuotas: entre 0% y 80%
    cuotas = np.round(ingresos * rng.uniform(0, 0.8, n), 1)
    endeudamiento = calcular_endeudamiento(ingresos, cuotas)

    # Política del banco
    aprobado = (
        (endeudamiento <= 40) & (ingresos >= 1) & (edad >= 21) & (edad <= 65)
    ).astype(int)

    # Excepciones: 5% de los casos no siguió la política
    excepciones = rng.random(n) < 0.05
    aprobado[excepciones] = 1 - aprobado[excepciones]

    return pd.DataFrame({
        "edad": edad,
        "ingresos": ingresos,
        "cuotas": cuotas,
        "endeudamiento": endeudamiento,
        "aprobado": aprobado,
    })


def como_tabla(cliente):
    """
    Convierte la solicitud {"edad", "ingresos", "cuotas"} en la tabla de una
    fila que reciben los modelos, con el endeudamiento ya calculado.
    """
    return pd.DataFrame([{
        "edad": cliente["edad"],
        "ingresos": cliente["ingresos"],
        "endeudamiento": float(calcular_endeudamiento(cliente["ingresos"], cliente["cuotas"])),
    }], columns=VARIABLES)


if __name__ == "__main__":
    df = generar_clientes()
    df.to_csv("clientes.csv", index=False)
    print(df.head(10))
    print(f"\n{len(df)} clientes, {df['aprobado'].mean():.0%} aprobados.")
    print("Guardado en clientes.csv")
