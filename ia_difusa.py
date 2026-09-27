"""
IA 3 - Lógica difusa, modelo Mamdani (Sesiones 3, 4 y 5).

Calcula el cupo del crédito en cuatro pasos:
    1. Fuzzificación:     ingresos y endeudamiento -> grados de pertenencia (sesión 3)
    2. Inferencia:        5 reglas SI ... ENTONCES, con min (Y) y max (O)  (sesión 4)
    3. Agregación:        cada salida se recorta a la fuerza de su regla
                          y todas se unen con max                         (sesión 4)
    4. Defuzzificación:   centroide -> un cupo exacto en millones          (sesión 5)
"""

import numpy as np

from datos import calcular_endeudamiento


def trapecio(x, a, b, c, d):
    """
    Función de pertenencia trapezoidal: sube de a hasta b, se queda en 1
    entre b y c, y baja de c hasta d.
    - Si b == c es el triángulo de la sesión 3.
    - Si a == b o c == d, un lado es vertical: sirve para los conjuntos de
      los extremos ("hombros"), que deben valer 1 hasta el borde.
    Acepta un número o un arreglo de NumPy.
    """
    x = np.asarray(x, dtype=float)
    y = np.where((x >= b) & (x <= c), 1.0, 0.0)
    if b > a:
        y = np.where((x > a) & (x < b), (x - a) / (b - a), y)
    if d > c:
        y = np.where((x > c) & (x < d), (d - x) / (d - c), y)
    return y


# -----------------------------------------------------------------------------
# Conjuntos difusos (definidos "por el experto": se pueden ajustar)
# -----------------------------------------------------------------------------
INGRESOS = {                 # millones al mes, de 0 a 15
    "bajo":  (0, 0, 1.5, 3.5),
    "medio": (2.5, 5, 5, 7.5),
    "alto":  (6, 9, 15, 15),
}
ENDEUDAMIENTO = {            # % de los ingresos que ya se va en cuotas, de 0 a 100
    "bajo":  (0, 0, 10, 25),
    "medio": (15, 30, 30, 45),
    "alto":  (35, 50, 100, 100),
}
CUPO = {                     # millones, de 0 a 50
    "bajo":     (0, 0, 2, 8),
    "medio":    (4, 12, 12, 20),
    "alto":     (14, 24, 24, 34),
    "muy alto": (28, 38, 50, 50),
}
UNIVERSO_CUPO = np.linspace(0, 50, 501)


def fuzzificar(valor, conjuntos, minimo, maximo):
    """Grado de pertenencia del valor a cada conjunto."""
    valor = min(max(valor, minimo), maximo)  # lo que se sale del rango se trata como el borde
    return {nombre: float(trapecio(valor, *p)) for nombre, p in conjuntos.items()}


def evaluar_reglas(gi, ge):
    """gi: grados de ingresos, ge: grados de endeudamiento."""
    return [
        {"regla": "SI ingresos es bajo O endeudamiento es alto ENTONCES cupo es bajo",
         "fuerza": max(gi["bajo"], ge["alto"]),        # O -> max
         "salida": "bajo"},
        {"regla": "SI ingresos es medio Y endeudamiento es medio ENTONCES cupo es medio",
         "fuerza": min(gi["medio"], ge["medio"]),      # Y -> min
         "salida": "medio"},
        {"regla": "SI ingresos es medio Y endeudamiento es bajo ENTONCES cupo es alto",
         "fuerza": min(gi["medio"], ge["bajo"]),
         "salida": "alto"},
        {"regla": "SI ingresos es alto Y endeudamiento es medio ENTONCES cupo es alto",
         "fuerza": min(gi["alto"], ge["medio"]),
         "salida": "alto"},
        {"regla": "SI ingresos es alto Y endeudamiento es bajo ENTONCES cupo es muy alto",
         "fuerza": min(gi["alto"], ge["bajo"]),
         "salida": "muy alto"},
    ]


def calcular_cupo(ingresos, cuotas):
    """Ejecuta el motor Mamdani completo y devuelve todos los pasos intermedios."""
    endeudamiento = float(calcular_endeudamiento(ingresos, cuotas))

    # 1. Fuzzificación
    gi = fuzzificar(ingresos, INGRESOS, 0, 15)
    ge = fuzzificar(endeudamiento, ENDEUDAMIENTO, 0, 100)

    # 2. Inferencia
    reglas = evaluar_reglas(gi, ge)

    # 3. Implicación (recortar cada salida a la fuerza de su regla) y agregación (max)
    agregado = np.zeros_like(UNIVERSO_CUPO)
    for r in reglas:
        recortado = np.fmin(trapecio(UNIVERSO_CUPO, *CUPO[r["salida"]]), r["fuerza"])
        agregado = np.fmax(agregado, recortado)

    # 4. Defuzzificación por centroide: Σ(x·μ) / Σμ
    area = agregado.sum()
    cupo = float(np.sum(UNIVERSO_CUPO * agregado) / area) if area > 0 else 0.0

    return {
        "endeudamiento": endeudamiento,
        "grados_ingresos": gi,
        "grados_endeudamiento": ge,
        "reglas": reglas,
        "agregado": agregado,
        "cupo": cupo,
    }


if __name__ == "__main__":
    for ingresos, cuotas in [(1.2, 0.1), (5.0, 1.5), (4.5, 0.4), (10.0, 0.5), (10.0, 6.0)]:
        r = calcular_cupo(ingresos, cuotas)
        print(f"Ingresos {ingresos:>4} | cuotas {cuotas:>3} | endeudamiento {r['endeudamiento']:5.1f}%"
              f" -> cupo {r['cupo']:5.1f} millones")
        for regla in r["reglas"]:
            if regla["fuerza"] > 0:
                print(f"      {regla['fuerza']:.2f}  {regla['regla']}")
