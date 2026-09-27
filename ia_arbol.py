"""
IA 1 - Árbol de decisión (Sesión 6).

Aprende las reglas del banco a partir de los casos históricos y, para cada
cliente nuevo, explica en forma SI ... ENTONCES por qué lo aprueba o lo rechaza.
Es el "sistema experto que escribe sus propias reglas".
"""

import math

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from datos import VARIABLES, como_tabla

CLASES = {0: "rechazado", 1: "aprobado"}


def entrenar(X, y):
    # entropy = usa Ganancia de Información, como en el taller analítico.
    # max_depth=4 evita que memorice el ruido; min_samples_leaf=5 evita
    # reglas basadas en uno o dos clientes raros.
    arbol = DecisionTreeClassifier(
        criterion="entropy", max_depth=4, min_samples_leaf=5, random_state=42
    )
    arbol.fit(X, y)
    return arbol


def predecir(arbol, cliente):
    """Devuelve (clase, confianza). clase: 1 aprueba, 0 rechaza."""
    probabilidades = arbol.predict_proba(como_tabla(cliente))[0]
    clase = int(np.argmax(probabilidades))
    return clase, float(probabilidades[clase])


# -----------------------------------------------------------------------------
# Explicaciones
# -----------------------------------------------------------------------------
def _clases_debajo(estructura, nodo):
    """Qué clases puede terminar diciendo el árbol a partir de este nodo."""
    if estructura.children_left[nodo] == -1:  # hoja
        return {int(np.argmax(estructura.value[nodo][0]))}
    return (_clases_debajo(estructura, estructura.children_left[nodo])
            | _clases_debajo(estructura, estructura.children_right[nodo]))


def _frase(variable, desde, hasta):
    """Convierte los límites de una variable en una frase: 'edad entre 21 y 65 años'."""
    if variable == "edad":
        # Los umbrales de edad quedan en x.5: "> 20.5" significa "21 o más"
        if desde is not None and hasta is not None:
            return f"edad entre {math.floor(desde) + 1} y {math.floor(hasta)} años"
        if hasta is not None:
            return f"edad de {math.floor(hasta)} años o menos"
        return f"edad de {math.floor(desde) + 1} años o más"

    if variable == "endeudamiento":
        if desde is not None and hasta is not None:
            return f"endeudamiento entre {desde:.1f}% y {hasta:.1f}%"
        if hasta is not None:
            return f"endeudamiento de {hasta:.1f}% o menos"
        return f"endeudamiento de más de {desde:.1f}%"

    if desde is not None and hasta is not None:
        return f"ingresos entre {desde:.2f} y {hasta:.2f} millones"
    if hasta is not None:
        return f"ingresos de {hasta:.2f} millones o menos"
    return f"ingresos de más de {desde:.2f} millones"


def _frases(limites, orden):
    return [_frase(v, *limites[v]) for v in orden]


def _bajar(limites, orden, variable, umbral, a_la_izquierda):
    """Anota la condición de una rama: izquierda es <= umbral, derecha es > umbral."""
    limites = {v: list(l) for v, l in limites.items()}
    desde, hasta = limites[variable]
    if a_la_izquierda:
        limites[variable][1] = umbral if hasta is None else min(hasta, umbral)
    else:
        limites[variable][0] = umbral if desde is None else max(desde, umbral)
    orden = orden if variable in orden else orden + [variable]
    return limites, orden


def explicar(arbol, cliente):
    """
    Recorre el árbol con los datos del cliente y devuelve las condiciones que
    cumplió, en el orden en que el árbol las preguntó. Se detiene apenas la
    decisión queda definida: si todas las ramas que siguen terminan en lo
    mismo, las preguntas restantes no cambian nada y no se muestran.
    """
    fila = como_tabla(cliente).iloc[0]
    t = arbol.tree_
    limites = {v: [None, None] for v in VARIABLES}
    orden = []
    nodo = 0

    while t.children_left[nodo] != -1 and len(_clases_debajo(t, nodo)) > 1:
        variable = VARIABLES[t.feature[nodo]]
        umbral = float(t.threshold[nodo])
        izquierda = fila[variable] <= umbral
        limites, orden = _bajar(limites, orden, variable, umbral, izquierda)
        nodo = t.children_left[nodo] if izquierda else t.children_right[nodo]

    return _frases(limites, orden)


def reglas_compactas(arbol):
    """
    Toda la base de conocimiento que aprendió el árbol, como una lista de
    reglas (condiciones, clase, número de clientes en que se basa).
    Las ramas que terminan todas en la misma clase se juntan en una sola regla.
    """
    t = arbol.tree_
    reglas = []

    def recorrer(nodo, limites, orden):
        clases = _clases_debajo(t, nodo)
        if len(clases) == 1:
            reglas.append((_frases(limites, orden), CLASES[clases.pop()], int(t.n_node_samples[nodo])))
            return
        variable = VARIABLES[t.feature[nodo]]
        umbral = float(t.threshold[nodo])
        recorrer(t.children_left[nodo], *_bajar(limites, orden, variable, umbral, True))
        recorrer(t.children_right[nodo], *_bajar(limites, orden, variable, umbral, False))

    recorrer(0, {v: [None, None] for v in VARIABLES}, [])
    return reglas


def todas_las_reglas(arbol):
    """El árbol completo tal como lo imprime scikit-learn (vista técnica)."""
    return export_text(arbol, feature_names=VARIABLES, class_names=["rechazado", "aprobado"])


def importancias(arbol):
    """Qué tanto usó cada variable para tomar sus decisiones (suman 1)."""
    return dict(zip(VARIABLES, arbol.feature_importances_))


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from datos import generar_clientes

    df = generar_clientes()
    entrenamiento, prueba = train_test_split(df, test_size=0.2, random_state=42, stratify=df["aprobado"])
    arbol = entrenar(entrenamiento[VARIABLES], entrenamiento["aprobado"])
    print(f"Exactitud con clientes que no vio: {arbol.score(prueba[VARIABLES], prueba['aprobado']):.0%}\n")

    print("Reglas que aprendió:")
    for condiciones, clase, n in reglas_compactas(arbol):
        print(f"   SI {' Y '.join(condiciones)} ENTONCES {clase}   ({n} clientes)")

    print()
    for edad, ingresos, cuotas in [(35, 4.5, 0.8), (30, 3.0, 2.0), (68, 6.0, 0.5), (19, 5.0, 0.5)]:
        cliente = {"edad": edad, "ingresos": ingresos, "cuotas": cuotas}
        clase, confianza = predecir(arbol, cliente)
        print(cliente, "->", CLASES[clase].upper(), f"({confianza:.0%})")
        print("      porque:", " Y ".join(explicar(arbol, cliente)))
