"""
Laboratorio Final - Starbucks Rewards
Actividad No. 2 - Grupo 12

App para decidir a que porcentaje de la base conviene mandar el cupon de $5.

Correr local con:  streamlit run app_streamlit.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Laboratorio Final - Starbucks Rewards", page_icon="*", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]

BASE_DIR = Path(__file__).resolve().parent

MODELOS = {
    "Logística": lambda: LogisticRegression(max_iter=1000),
    "Árbol sin límite": lambda: DecisionTreeClassifier(random_state=42),
    "Árbol max_depth=2": lambda: DecisionTreeClassifier(max_depth=2, random_state=42),
    "Árbol max_depth=4": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=200, random_state=42),
}


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "uplift_campaign.csv")


@st.cache_data
def load_full():
    ruta = BASE_DIR / "uplift_campaign_FULL.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data(show_spinner="Entrenando los modelos...")
def entrenar_t_learner(df: pd.DataFrame, nombre_modelo: str, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed,
                                   stratify=df["treatment_group"])
    constructor = MODELOS[nombre_modelo]

    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = constructor().fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control = constructor().fit(train_c[FEATURES], train_c["responded_60d"])

    train_churn = train.copy()
    train_churn["churned"] = 1 - train_churn["responded_60d"]
    modelo_riesgo = LogisticRegression(max_iter=1000).fit(train_churn[FEATURES], train_churn["churned"])

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    test["prob_churn"] = modelo_riesgo.predict_proba(X_test)[:, 1]
    return test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values

    acum_t = np.cumsum(np.where(es_tratado, respondio, 0))
    acum_c = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_t = np.cumsum(es_tratado)
    n_c = np.cumsum(~es_tratado)

    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt, nc = n_t[i - 1], n_c[i - 1]
        tasa_t = acum_t[i - 1] / nt if nt > 0 else 0
        tasa_c = acum_c[i - 1] / nc if nc > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)
    area_azar = np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n


def tabla_de_politicas(test, paso):
    filas = []
    for pct in range(paso, 101, paso):
        n_c = int(len(test) * pct / 100)
        idx = test.sort_values("uplift_estimado", ascending=False).head(n_c).index
        valor, n = valor_de_la_politica(test, idx)
        filas.append({"% contactado": pct, "Clientes": n,
                      "Valor por cliente ($)": round(valor, 2),
                      "Valor total ($)": round(valor * n, 2)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- encabezado
st.title("Laboratorio Final - Starbucks Rewards")
st.caption(
    "Esta app responde una sola pregunta: a qué porcentaje de la base conviene mandarle el "
    "cupón de \\$5. Ordena a los clientes por el uplift estimado del T-learner y muestra cuánto "
    "dinero deja cada corte, medido contra el grupo de control del experimento."
)

df = load_data()
full = load_full()

# ---------------------------------------------------------------- controles
with st.sidebar:
    st.header("Parámetros")
    nombre_modelo = st.selectbox(
        "Modelo base del T-learner", list(MODELOS.keys()), index=4,
        help="El modelo que se entrena dos veces: uno con los tratados y otro con el control.")
    pct_contactar = st.slider(
        "% de la base a contactar", min_value=5, max_value=100, value=20, step=5,
        help="Se contactan los clientes con mayor uplift estimado.")
    paso_tabla = st.radio("Granularidad de la tabla", [5, 10], index=0,
                          format_func=lambda x: f"cada {x}%", horizontal=True)
    st.divider()
    test_size = st.slider("Tamaño del conjunto de prueba", 0.20, 0.50, 0.30, 0.05)
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=9999, value=42, step=1)
    st.divider()
    st.caption("Costo del cupón: \\$5 si el cliente compra. Envío: \\$0.05 por cliente tratado.")

test = entrenar_t_learner(df, nombre_modelo, test_size, int(seed))

# ---------------------------------------------------------------- indicadores
xs_u, ys_u = curva_qini(test, "uplift_estimado")
xs_r, ys_r = curva_qini(test, "prob_churn")
qini_u = coeficiente_qini(xs_u, ys_u, len(test))
qini_r = coeficiente_qini(xs_r, ys_r, len(test))

tabla = tabla_de_politicas(test, paso_tabla)
optimo = tabla.loc[tabla["Valor total ($)"].idxmax()]

n_sel = int(len(test) * pct_contactar / 100)
idx_sel = test.sort_values("uplift_estimado", ascending=False).head(n_sel).index
valor_sel, _ = valor_de_la_politica(test, idx_sel)
valor_total_sel = valor_sel * n_sel

valor_todos, n_todos = valor_de_la_politica(test, test.index)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Qini del T-learner", f"{qini_u:+.3f}", f"{qini_u - qini_r:+.3f} vs. riesgo")
c2.metric(f"Valor total al {pct_contactar}%", f"${valor_total_sel:,.0f}",
          f"${valor_total_sel - optimo['Valor total ($)']:,.0f} vs. óptimo")
c3.metric("Valor por cliente", f"${valor_sel:,.2f}")
c4.metric("Óptimo del modelo", f"{int(optimo['% contactado'])}%", f"${optimo['Valor total ($)']:,.0f}")

if pct_contactar == int(optimo["% contactado"]):
    st.success(
        f"El {pct_contactar}% es el corte óptimo de este modelo: deja "
        f"\\${valor_total_sel:,.2f} sobre {n_sel:,} clientes contactados.")
elif valor_total_sel > 0:
    st.info(
        f"Contactar al {pct_contactar}% deja \\${valor_total_sel:,.2f}. El óptimo está en "
        f"{int(optimo['% contactado'])}% con \\${optimo['Valor total ($)']:,.2f}, una diferencia de "
        f"\\${optimo['Valor total ($)'] - valor_total_sel:,.2f}.")
else:
    st.error(
        f"Contactar al {pct_contactar}% pierde \\${abs(valor_total_sel):,.2f}. A ese nivel ya se "
        f"está mandando cupón a clientes que iban a comprar de todos modos.")

st.divider()

# ---------------------------------------------------------------- graficas
col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Curva Qini del T-learner")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs_u, ys_u, color=GREEN, linewidth=3, label=f"T-learner ({nombre_modelo}) · Qini={qini_u:+.3f}")
    ax.plot(xs_r, ys_r, color=ORANGE, linewidth=3, label=f"Riesgo/churn · Qini={qini_r:+.3f}")
    ax.plot([0, 1], [0, ys_u[-1]], color=GRAY, linestyle="--", label="Al azar")
    ax.axvline(pct_contactar / 100, color="#11241C", linestyle=":", linewidth=2,
               label=f"Corte actual: {pct_contactar}%")
    ax.set_xlabel("% de la base contactada")
    ax.set_ylabel("Respuestas incrementales acumuladas")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Entre más arriba de la diagonal gris, mejor ordena el modelo. El Qini es el área "
        "entre la curva y esa diagonal.")

with col_der:
    st.subheader("Dónde deja de convenir contactar")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    colores = [GREEN if v >= 0 else ORANGE for v in tabla["Valor total ($)"]]
    ax2.bar(tabla["% contactado"], tabla["Valor total ($)"], width=paso_tabla * 0.75, color=colores)
    ax2.axhline(0, color="#11241C", linewidth=1)
    ax2.axvline(pct_contactar, color="#11241C", linestyle=":", linewidth=2)
    ax2.set_xlabel("% de la base contactada")
    ax2.set_ylabel("Valor neto total ($)")
    ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption(
        f"Contactar a toda la base pierde \\${abs(valor_todos * n_todos):,.0f}. "
        "Las barras naranjas son cortes que destruyen valor.")

st.divider()

# ---------------------------------------------------------------- tabla
st.subheader("Tabla de políticas")
st.caption(
    "Cada fila es una decisión posible: contactar a ese porcentaje de la base, ordenada de "
    "mayor a menor uplift estimado. La fila resaltada es la que más dinero deja.")


def resaltar_optimo(fila):
    if fila["% contactado"] == optimo["% contactado"]:
        return ["background-color: #00704A; color: white; font-weight: bold"] * len(fila)
    if fila["% contactado"] == pct_contactar:
        return ["background-color: #E8F0ED"] * len(fila)
    return [""] * len(fila)


st.dataframe(
    tabla.style.apply(resaltar_optimo, axis=1).format({
        "Valor por cliente ($)": "${:,.2f}",
        "Valor total ($)": "${:,.2f}",
        "Clientes": "{:,}",
    }),
    use_container_width=True, hide_index=True, height=min(430, 40 + 36 * len(tabla)))

# ---------------------------------------------------------------- revelacion
if full is not None:
    st.divider()
    st.subheader("Qué tipo de cliente está recibiendo el cupón")
    st.caption(
        "Esta sección usa la columna oculta del experimento, que en una campaña real no existe. "
        "Sirve para verificar si el modelo está eligiendo a la gente correcta.")

    ft = full[full["customer_id"].isin(test["customer_id"])].merge(
        test[["customer_id", "uplift_estimado", "prob_churn"]], on="customer_id")
    sel = ft.nlargest(n_sel, "uplift_estimado")

    comp = pd.DataFrame({
        f"{pct_contactar}% por uplift (%)": (sel["true_type"].value_counts(normalize=True) * 100).round(1),
        "Base completa (%)": (ft["true_type"].value_counts(normalize=True) * 100).round(1),
        "Valor real de contacto ($)": ft.groupby("true_type")["true_value_of_contact"].mean().round(2),
    }).fillna(0)
    orden_tipos = ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]
    comp = comp.reindex([t for t in orden_tipos if t in comp.index])

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.dataframe(comp, use_container_width=True)
    with col_b:
        corr = sel["uplift_estimado"].corr(sel["true_uplift_prob"])
        corr_total = ft["uplift_estimado"].corr(ft["true_uplift_prob"])
        pers = (sel["true_type"] == "persuadable").mean() * 100
        st.metric("Persuadables en el grupo contactado", f"{pers:.1f}%",
                  f"{pers - (ft['true_type'] == 'persuadable').mean() * 100:+.1f} pp vs. base")
        st.metric("Correlación con el uplift real", f"{corr_total:.3f}")
        st.metric("Valor real promedio del grupo", f"${sel['true_value_of_contact'].mean():+.2f}")
