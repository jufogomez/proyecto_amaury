"""
Asesor de crédito - Proyecto final de Inteligencia Artificial.

Un comité de tres IAs revisa cada solicitud de crédito:
    - El árbol de decisión decide y explica su regla      (ia_arbol.py)
    - El KNN decide según los 5 clientes más parecidos    (ia_knn.py)
    - La lógica difusa calcula el cupo del crédito        (ia_difusa.py)

Si el árbol y el KNN coinciden, esa es la decisión. Si no coinciden, el
caso pasa a revisión de un analista humano.

Para ejecutarla:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

import graficas
import ia_arbol
import ia_difusa
import ia_knn
from datos import VARIABLES, calcular_endeudamiento, generar_clientes

st.set_page_config(page_title="Asesor de crédito", page_icon="🏦", layout="wide")

# Estilo de la tarjeta del dictamen (el resto usa el tema de .streamlit/config.toml)
st.markdown("""<style>
.dictamen {display: flex; align-items: center; gap: 2.25rem; padding: 1.5rem 1.75rem;
  margin: .75rem 0 1.75rem; background: #FFFFFF; border: 1px solid #D5DCE4; border-radius: 6px;}
.dictamen .sello {font-family: "IBM Plex Serif", Georgia, serif; font-size: 2.1rem; font-weight: 600;
  line-height: 1; padding: .55rem 1.1rem .6rem; border: 4px double currentColor; border-radius: 6px;
  transform: rotate(-4deg); white-space: nowrap;}
.dictamen.aprobado .sello {color: #1E7B5C;}
.dictamen.rechazado .sello {color: #B23A34;}
.dictamen.revision .sello {color: #A8770F;}
.dictamen .etiqueta {font-size: .95rem; color: #52606E;}
.dictamen .cifra {font-size: 2.3rem; font-weight: 600; color: #1C2B3A; line-height: 1.15;
  font-variant-numeric: tabular-nums;}
.dictamen .nota {margin-top: .45rem; color: #3A4856; max-width: 64ch; line-height: 1.5;}
@media (max-width: 640px) {.dictamen {flex-direction: column; align-items: flex-start; gap: 1.1rem;}}
</style>""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Entrenamiento: se hace una sola vez y queda guardado en caché
# -----------------------------------------------------------------------------
@st.cache_resource
def preparar_comite():
    historico = generar_clientes()
    # 80% para aprender, 20% para medir qué tan bien aciertan con casos nuevos
    entrenamiento, prueba = train_test_split(
        historico, test_size=0.2, random_state=42, stratify=historico["aprobado"]
    )
    X, y = entrenamiento[VARIABLES], entrenamiento["aprobado"]
    arbol = ia_arbol.entrenar(X, y)
    knn = ia_knn.entrenar(X, y)
    return {
        "historico": historico,
        "entrenamiento": entrenamiento,
        "arbol": arbol,
        "knn": knn,
        "exactitud_arbol": arbol.score(prueba[VARIABLES], prueba["aprobado"]),
        "exactitud_knn": knn.score(prueba[VARIABLES], prueba["aprobado"]),
        "n_prueba": len(prueba),
    }


def pesos(millones):
    """3.25 -> '$3.300.000' (redondeado a cien mil, con puntos de miles)."""
    return "$" + f"{round(millones * 1e6, -5):,.0f}".replace(",", ".")


def mostrar(figura):
    st.pyplot(figura)
    plt.close(figura)


comite = preparar_comite()

# -----------------------------------------------------------------------------
# Solicitud
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Solicitud")
    st.caption("Cambia cualquier valor y el comité vuelve a evaluar al instante.")
    edad = st.slider("Edad", 18, 75, 35)
    ingresos = st.slider("Ingresos al mes (millones de pesos)", 0.5, 15.0, 4.5, 0.1)
    cuotas = st.slider("Cuotas que ya paga al mes (millones de pesos)", 0.0, 10.0, 0.8, 0.1)
    endeudamiento = float(calcular_endeudamiento(ingresos, cuotas))
    st.metric("Endeudamiento", f"{endeudamiento:.1f}%",
              help="Qué parte de sus ingresos ya se va en cuotas. Se calcula solo.")

cliente = {"edad": edad, "ingresos": ingresos, "cuotas": cuotas}

# -----------------------------------------------------------------------------
# Las tres IAs evalúan la solicitud
# -----------------------------------------------------------------------------
voto_arbol, confianza_arbol = ia_arbol.predecir(comite["arbol"], cliente)
condiciones = ia_arbol.explicar(comite["arbol"], cliente)
voto_knn, votos_a_favor, vecinos = ia_knn.consultar(comite["knn"], comite["entrenamiento"], cliente)
difusa = ia_difusa.calcular_cupo(ingresos, cuotas)
cupo = difusa["cupo"]

# Decisión del comité
if voto_arbol and voto_knn:
    estado, sello, etiqueta, cifra = "aprobado", "Aprobado", "Cupo aprobado", pesos(cupo)
    nota = (f"El árbol de decisión aprueba con {confianza_arbol:.0%} de confianza y "
            f"{votos_a_favor} de 5 clientes parecidos fueron aprobados. "
            "La lógica difusa calculó el cupo.")
elif not voto_arbol and not voto_knn:
    estado, sello, etiqueta, cifra = "rechazado", "Rechazado", "", "Sin cupo"
    parecidos = ("ninguno de los 5 clientes parecidos fue aprobado" if votos_a_favor == 0
                 else f"solo {votos_a_favor} de 5 clientes parecidos fueron aprobados")
    nota = (f"El árbol de decisión rechaza con {confianza_arbol:.0%} de confianza y {parecidos}. "
            f"Motivo según el árbol: {' y '.join(condiciones)}.")
else:
    estado, sello, etiqueta, cifra = "revision", "En revisión", "Cupo sugerido", pesos(cupo)
    if voto_arbol:
        desacuerdo = (f"el árbol de decisión aprueba, pero {5 - votos_a_favor} de "
                      "5 clientes parecidos fueron rechazados")
    else:
        desacuerdo = (f"{votos_a_favor} de 5 clientes parecidos fueron aprobados, "
                      "pero el árbol de decisión rechaza")
    nota = (f"No hay acuerdo: {desacuerdo}. Un analista debe revisar el caso; "
            "si lo aprueba, este es el cupo sugerido.")

# -----------------------------------------------------------------------------
# Página principal
# -----------------------------------------------------------------------------
st.title("Asesor de crédito")
st.write("Tres inteligencias artificiales revisan cada solicitud: el árbol de decisión y "
         "el KNN deciden si se aprueba, y la lógica difusa calcula el cupo.")

st.markdown(
    f'<div class="dictamen {estado}"><div class="sello">{sello}</div><div>'
    f'<div class="etiqueta">{etiqueta}</div><div class="cifra">{cifra}</div>'
    f'<div class="nota">{nota}</div></div></div>',
    unsafe_allow_html=True,
)

pestana_arbol, pestana_knn, pestana_difusa, pestana_datos = st.tabs(
    ["Árbol de decisión", "Vecinos (KNN)", "Lógica difusa", "Datos del banco"]
)

# --- IA 1: árbol de decisión --------------------------------------------------
with pestana_arbol:
    st.subheader("Decide y explica su regla")
    st.write(f"El árbol **{'aprueba' if voto_arbol else 'rechaza'}** la solicitud con "
             f"{confianza_arbol:.0%} de confianza. Esta es la regla que se aplicó:")
    regla = "SI        " + "\nY         ".join(condiciones)
    regla += f"\nENTONCES  {ia_arbol.CLASES[voto_arbol]}"
    st.code(regla, language=None)

    izquierda, derecha = st.columns([3, 2], gap="large")
    with izquierda:
        st.markdown("**Todas las reglas que aprendió por su cuenta**")
        for conds, clase, n in ia_arbol.reglas_compactas(comite["arbol"]):
            st.markdown(f"- SI {' Y '.join(conds)} ENTONCES **{clase}** ({n} clientes)")
    with derecha:
        st.markdown("**Qué tanto usó cada variable**")
        mostrar(graficas.importancias(ia_arbol.importancias(comite["arbol"])))

    st.caption(f"Acierta el {comite['exactitud_arbol']:.0%} de los {comite['n_prueba']} "
               "clientes que no vio durante el entrenamiento.")
    with st.expander("Ver el árbol completo (vista técnica de scikit-learn)"):
        st.code(ia_arbol.todas_las_reglas(comite["arbol"]), language=None)

# --- IA 2: KNN ----------------------------------------------------------------
with pestana_knn:
    st.subheader("Compara con los 5 clientes más parecidos")
    st.write(f"**{votos_a_favor} de 5** vecinos fueron aprobados, así que el KNN "
             f"**{'aprueba' if voto_knn else 'rechaza'}** la solicitud.")

    izquierda, derecha = st.columns([3, 2], gap="large")
    with izquierda:
        mostrar(graficas.vecinos(comite["entrenamiento"],
                                 {"ingresos": ingresos, "endeudamiento": endeudamiento},
                                 vecinos))
    with derecha:
        st.dataframe(
            vecinos.rename(columns={
                "edad": "Edad", "ingresos": "Ingresos", "cuotas": "Cuotas",
                "endeudamiento": "Endeud. (%)", "resultado": "Resultado",
                "distancia": "Distancia",
            }),
            hide_index=True,
        )
        st.caption("Ingresos y cuotas en millones de pesos. La distancia se mide con las "
                   "columnas normalizadas, para que ninguna pese más solo por tener números grandes.")

    st.caption("Los vecinos se buscan con las tres variables, incluida la edad, así que en esta "
               "vista de dos variables no siempre se ven como los más cercanos. "
               f"El KNN acierta el {comite['exactitud_knn']:.0%} de los clientes que no vio.")

# --- IA 3: lógica difusa --------------------------------------------------------
with pestana_difusa:
    st.subheader("Calcula el cupo con reglas difusas")
    st.write(f"Con ingresos de {ingresos:.1f} millones y un endeudamiento de {endeudamiento:.1f}%, "
             f"el sistema difuso calcula un cupo de **{pesos(cupo)}**.")
    mostrar(graficas.difusa(difusa, ingresos))

    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        st.markdown("**1. Fuzzificación:** qué tanto pertenece a cada conjunto")
        gi, ge = difusa["grados_ingresos"], difusa["grados_endeudamiento"]
        st.dataframe(pd.DataFrame([
            {"Variable": "Ingresos", "Valor": f"{ingresos:.1f} millones",
             "Bajo": gi["bajo"], "Medio": gi["medio"], "Alto": gi["alto"]},
            {"Variable": "Endeudamiento", "Valor": f"{endeudamiento:.1f}%",
             "Bajo": ge["bajo"], "Medio": ge["medio"], "Alto": ge["alto"]},
        ]).round(2), hide_index=True)
    with derecha:
        st.markdown("**2. Inferencia:** fuerza de cada regla (Y = mínimo, O = máximo)")
        st.dataframe(
            pd.DataFrame([{"Regla": r["regla"], "Fuerza": r["fuerza"]} for r in difusa["reglas"]]),
            hide_index=True,
            column_config={"Fuerza": st.column_config.ProgressColumn(
                "Fuerza", min_value=0.0, max_value=1.0, format="%.2f")},
        )

    st.markdown("**3. Agregación:** cada conjunto de salida se recorta a la fuerza de su regla "
                "y todos se unen con el máximo (el área azul de la gráfica).")
    st.markdown(f"**4. Defuzzificación:** el centroide de esa área da el cupo exacto: "
                f"{cupo:.1f} millones, que se redondean a {pesos(cupo)}.")
    if estado == "rechazado":
        st.info("Como el comité rechazó la solicitud, este cupo no se ofrece.")

# --- Datos ----------------------------------------------------------------------
with pestana_datos:
    historico = comite["historico"]
    st.subheader("Los casos históricos con que aprendieron")
    st.write(
        "Son datos simulados con la política del banco: endeudamiento de máximo 40%, "
        "ingresos de al menos 1.5 millones y edad entre 21 y 65 años, más un 5% de "
        "excepciones al azar. Las IAs no conocen esta política; el árbol la descubrió por "
        "su cuenta, como puedes comparar con sus reglas en la primera pestaña."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes", len(historico))
    c2.metric("Aprobados", f"{historico['aprobado'].mean():.0%}")
    c3.metric("Para entrenar / para evaluar",
              f"{len(comite['entrenamiento'])} / {comite['n_prueba']}")
    st.dataframe(
        historico.assign(aprobado=historico["aprobado"].map({1: "Sí", 0: "No"})).rename(columns={
            "edad": "Edad", "ingresos": "Ingresos (millones)", "cuotas": "Cuotas (millones)",
            "endeudamiento": "Endeudamiento (%)", "aprobado": "Aprobado",
        }),
        hide_index=True,
    )
    st.download_button("Descargar los datos en CSV", historico.to_csv(index=False).encode("utf-8"),
                       file_name="clientes.csv", mime="text/csv")

st.caption("Proyecto académico de Inteligencia Artificial. Los datos son simulados y las "
           "decisiones no corresponden a ninguna entidad financiera real.")
