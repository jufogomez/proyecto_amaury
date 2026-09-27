"""
IA 2 - K vecinos más cercanos, KNN (Sesión 9).

Busca a los 5 clientes históricos más parecidos al solicitante y los pone a
votar: "dime con quién andas y te diré quién eres".

Antes de medir distancias se normalizan las columnas con StandardScaler.
Sin ese paso, una diferencia de 10 puntos de endeudamiento pesaría muchísimo
más que una diferencia de 2 millones de ingresos, solo porque los números son
más grandes (la trampa que vimos en la sesión 9).
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datos import VARIABLES, como_tabla

K = 5  # impar, para que nunca haya empate en la votación


def entrenar(X, y):
    modelo = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    modelo.fit(X, y)  # KNN no aprende una ecuación: solo memoriza los datos
    return modelo


def consultar(modelo, historico, cliente):
    """
    Devuelve (clase, votos_a_favor, tabla_de_vecinos).
    historico: la tabla completa de clientes con la que se entrenó el modelo.
    """
    normalizador = modelo.named_steps["standardscaler"]
    knn = modelo.named_steps["kneighborsclassifier"]

    distancias, posiciones = knn.kneighbors(normalizador.transform(como_tabla(cliente)))
    posiciones = posiciones[0]

    vecinos = historico.iloc[posiciones][["edad", "ingresos", "cuotas", "endeudamiento"]].copy()
    votos = historico["aprobado"].iloc[posiciones]
    vecinos["resultado"] = votos.map({1: "Aprobado", 0: "Rechazado"}).values
    vecinos["distancia"] = distancias[0].round(2)

    votos_a_favor = int(votos.sum())
    clase = int(votos_a_favor > K / 2)
    return clase, votos_a_favor, vecinos.reset_index(drop=True)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from datos import generar_clientes

    df = generar_clientes()
    entrenamiento, prueba = train_test_split(df, test_size=0.2, random_state=42, stratify=df["aprobado"])
    modelo = entrenar(entrenamiento[VARIABLES], entrenamiento["aprobado"])
    print(f"Exactitud con clientes que no vio: {modelo.score(prueba[VARIABLES], prueba['aprobado']):.0%}\n")

    clase, votos, vecinos = consultar(modelo, entrenamiento, {"edad": 35, "ingresos": 4.5, "cuotas": 0.8})
    print(vecinos)
    print(f"\n{votos} de {K} vecinos fueron aprobados ->", "APRUEBA" if clase else "RECHAZA")
