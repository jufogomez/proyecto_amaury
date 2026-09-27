"""
Gráficas de la aplicación (matplotlib).
Cada función devuelve una figura lista para mostrar con st.pyplot().
"""
#suave

import matplotlib

matplotlib.use("Agg")  # dibuja sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np

import ia_difusa

TINTA = "#1C2B3A"
GRIS = "#8A96A3"
VERDE = "#1E7B5C"
ROJO = "#B23A34"
AZUL = "#2B5F8A"
AZULES = ["#9CBCD8", "#5B8FBF", "#2B5F8A", "#163A58"]  # de "bajo" a "muy alto"


def _limpiar(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRIS)
    ax.tick_params(colors=TINTA, labelsize=9)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def vecinos(historico, cliente, tabla_vecinos):
    """Clientes históricos (ingresos contra endeudamiento), la solicitud y sus vecinos."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    aprobados = historico["aprobado"] == 1

    ax.scatter(historico.loc[aprobados, "ingresos"], historico.loc[aprobados, "endeudamiento"],
               s=16, color=VERDE, alpha=0.35, label="Aprobados")
    ax.scatter(historico.loc[~aprobados, "ingresos"], historico.loc[~aprobados, "endeudamiento"],
               s=16, color=ROJO, alpha=0.35, label="Rechazados")
    ax.scatter(tabla_vecinos["ingresos"], tabla_vecinos["endeudamiento"],
               s=190, facecolors="none", edgecolors=TINTA, linewidths=1.4, label="Sus 5 vecinos")
    ax.scatter([cliente["ingresos"]], [cliente["endeudamiento"]], s=320, marker="*",
               color=AZUL, edgecolors="white", linewidths=1.2, zorder=5, label="La solicitud")

    ax.set_xlabel("Ingresos al mes (millones)", color=TINTA)
    ax.set_ylabel("Endeudamiento (%)", color=TINTA)
    ax.set_xlim(0, max(15.5, cliente["ingresos"] + 0.5))
    ax.set_ylim(-3, max(85, cliente["endeudamiento"] + 5))
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92, edgecolor="none")
    _limpiar(ax)
    fig.tight_layout()
    return fig


def difusa(resultado, ingresos):
    """Las funciones de pertenencia con los valores del cliente y la salida agregada."""
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.3))

    entradas = [
        (ejes[0], ia_difusa.INGRESOS, np.linspace(0, 15, 301), min(ingresos, 15),
         "Ingresos al mes (millones)"),
        (ejes[1], ia_difusa.ENDEUDAMIENTO, np.linspace(0, 100, 301),
         min(resultado["endeudamiento"], 100), "Endeudamiento (%)"),
    ]
    for ax, conjuntos, x, valor, titulo in entradas:
        for color, (nombre, p) in zip(AZULES, conjuntos.items()):
            ax.plot(x, ia_difusa.trapecio(x, *p), color=color, linewidth=2, label=nombre)
        ax.axvline(valor, color=TINTA, linestyle="--", linewidth=1.2)
        ax.text(valor, 1.1, f"{valor:.1f}", ha="center", fontsize=8, color=TINTA,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_title(titulo, fontsize=10, color=TINTA)
        ax.set_ylim(-0.03, 1.2)
        ax.legend(fontsize=8, frameon=False, loc="center right")
        _limpiar(ax)

    # Salida: conjuntos originales (punteados), recortes por regla y centroide
    ax = ejes[2]
    x = ia_difusa.UNIVERSO_CUPO
    for color, (nombre, p) in zip(AZULES, ia_difusa.CUPO.items()):
        ax.plot(x, ia_difusa.trapecio(x, *p), color=color, linewidth=1, linestyle=":")
    ax.fill_between(x, resultado["agregado"], color=AZUL, alpha=0.28, label="Salida agregada")
    ax.plot(x, resultado["agregado"], color=AZUL, linewidth=1.6)
    ax.axvline(resultado["cupo"], color=TINTA, linewidth=2, label="Centroide")
    ax.set_title(f"Cupo: centroide en {resultado['cupo']:.1f} millones", fontsize=10, color=TINTA)
    ax.set_ylim(-0.03, 1.2)
    ax.legend(fontsize=8, frameon=True, framealpha=0.92, edgecolor="none", loc="best")
    _limpiar(ax)

    fig.tight_layout()
    return fig


def importancias(pesos):
    """Barras horizontales con cuánto usó el árbol cada variable."""
    nombres = {"edad": "Edad", "ingresos": "Ingresos", "endeudamiento": "Endeudamiento"}
    orden = sorted(pesos, key=pesos.get)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.barh([nombres[v] for v in orden], [pesos[v] for v in orden], color=AZUL, height=0.55)
    for i, v in enumerate(orden):
        ax.text(pesos[v] + 0.01, i, f"{pesos[v]:.0%}", va="center", fontsize=9, color=TINTA)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    _limpiar(ax)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig
