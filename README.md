# Asesor de crédito

Aplicación web que evalúa solicitudes de crédito con un comité de tres inteligencias artificiales. Dos de ellas deciden si el crédito se aprueba, y la tercera calcula el cupo exacto. Cada IA muestra cómo llegó a su respuesta.

Proyecto final de Inteligencia Artificial.

## Las tres IAs

| IA | Tema del curso | Qué hace en la aplicación | Archivo |
|---|---|---|---|
| Árbol de decisión | Sesión 6 | Aprueba o rechaza, y explica la regla SI ... ENTONCES que aplicó | `ia_arbol.py` |
| K vecinos más cercanos (KNN) | Sesión 9 | Busca los 5 clientes históricos más parecidos y los pone a votar | `ia_knn.py` |
| Lógica difusa (Mamdani) | Sesiones 3, 4 y 5 | Calcula el cupo con reglas difusas y lo convierte en un valor exacto con el centroide | `ia_difusa.py` |

## Cómo decide el comité

1. El usuario escribe su edad, sus ingresos al mes y las cuotas que ya paga. La aplicación calcula su endeudamiento, es decir, qué porcentaje de los ingresos ya se va en cuotas.
2. El árbol de decisión y el KNN dan cada uno su voto: aprobar o rechazar.
3. Si los dos aprueban, el crédito se aprueba con el cupo que calculó la lógica difusa.
4. Si los dos rechazan, el crédito se rechaza, y el árbol explica el motivo.
5. Si no se ponen de acuerdo, el caso queda "En revisión" para que lo decida un analista humano, y se muestra el cupo sugerido.

## Cómo ejecutarla

Se necesita Python 3.10 o más reciente.

```bash
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abre sola en el navegador, en `http://localhost:8501`. Hay que ejecutar el comando desde la carpeta del proyecto para que cargue el tema visual de `.streamlit/config.toml`.

Cada IA también se puede probar por separado en la consola, lo cual sirve para explicarlas una por una:

```bash
python ia_arbol.py
python ia_knn.py
python ia_difusa.py
python datos.py      # además guarda los datos en clientes.csv
```

## Estructura del proyecto

```
asesor_credito/
├── app.py                 Interfaz web (Streamlit): formulario, dictamen y una pestaña por IA
├── datos.py               Genera los 400 clientes históricos simulados
├── ia_arbol.py            IA 1: árbol de decisión y sus explicaciones
├── ia_knn.py              IA 2: K vecinos más cercanos
├── ia_difusa.py           IA 3: motor difuso Mamdani con defuzzificación por centroide
├── graficas.py            Gráficas de matplotlib que usa la interfaz
├── requirements.txt       Librerías necesarias
└── .streamlit/
    └── config.toml        Colores y fuentes de la aplicación
```

## Los datos

Son 400 clientes simulados, generados con la política del banco:

- el endeudamiento no puede pasar del 40%,
- hay que ganar al menos 1.5 millones al mes,
- hay que tener entre 21 y 65 años.

Además, un 5% de las decisiones se voltea al azar, para imitar las excepciones de la vida real. De los 400 clientes, 185 fueron aprobados (46%).

Las IAs no conocen esta política. Aprenden con 320 clientes y se evalúan con los otros 80, que nunca vieron:

- el árbol de decisión acierta 78 de 80 (97.5%),
- el KNN acierta 69 de 80 (86%).

Lo más interesante es que el árbol redescubrió la política del banco por su cuenta. Estas son las reglas que aprendió:

```
SI endeudamiento de 40.1% o menos Y edad de 20 años o menos                              ENTONCES rechazado
SI endeudamiento de 40.1% o menos Y edad entre 21 y 65 años Y ingresos de 1.45 millones o menos ENTONCES rechazado
SI endeudamiento de 40.1% o menos Y edad entre 21 y 65 años Y ingresos de más de 1.45 millones  ENTONCES aprobado
SI endeudamiento de 40.1% o menos Y edad de 66 años o más                                ENTONCES rechazado
SI endeudamiento de más de 40.1%                                                         ENTONCES rechazado
```

El endeudamiento no lo escribe el usuario: la aplicación lo calcula a partir de los ingresos y las cuotas. Esto se llama ingeniería de características. Con ingresos y cuotas por separado, el árbol solo acertaba cerca del 85%, porque tenía que aproximar una proporción con preguntas sueltas.

## Casos para la sustentación

Estos casos ya están probados. Se escriben en el formulario de la izquierda.

| Caso | Edad | Ingresos | Cuotas | Resultado |
|---|---|---|---|---|
| Cliente ideal | 35 | 4.5 | 0.8 | Aprobado, cupo de $21.100.000 |
| Ingresos altos | 45 | 12.0 | 1.0 | Aprobado, cupo de $41.300.000 |
| Endeudamiento alto | 30 | 3.0 | 2.0 | Rechazado: endeudamiento de 66.7% |
| Muy joven | 19 | 5.0 | 0.5 | En revisión: el árbol rechaza por edad, pero 3 de 5 vecinos fueron aprobados |
| Mayor de 65 | 68 | 6.0 | 0.5 | En revisión: el mismo desacuerdo, por el otro extremo de edad |
| Justo en el límite | 25 | 3.0 | 1.2 | En revisión: endeudamiento de 40%; el árbol aprueba y 3 de 5 vecinos fueron rechazados |

Los casos "En revisión" muestran por qué conviene tener más de una IA. El árbol aprendió los cortes exactos de la política, como la edad máxima de 65 años. En cambio, el KNN solo ve a los clientes parecidos, y cerca de un límite sus vecinos quedan mezclados. Cuando no coinciden, lo prudente es que decida una persona.

## Qué se puede ajustar

- `datos.py`: la cantidad de clientes, la política del banco o el porcentaje de excepciones.
- `ia_arbol.py`: la profundidad máxima del árbol (`max_depth`).
- `ia_knn.py`: el número de vecinos (`K`, que debe ser impar).
- `ia_difusa.py`: los conjuntos difusos y las reglas del cupo.

Si se cambian los datos, las reglas y los porcentajes de acierto también cambian.

---

Los datos son simulados y las decisiones no corresponden a ninguna entidad financiera real.
