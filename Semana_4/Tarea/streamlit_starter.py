"""
Laboratorio de Recomendacion — app de Streamlit (Actividad 3, Grupo 12).

App para que alguien de negocio explore los recomendadores: elegir un cliente y un metodo,
ver sus 3 recomendaciones con el "porque", y comparar los metodos (hit rate y cobertura),
con controles interactivos.

Correr con:  streamlit run streamlit_starter.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"

# Rutas relativas a este archivo, para que funcione igual en local y en Streamlit Cloud
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_data
def cargar():
    catalogo = pd.read_csv(os.path.join(BASE, "catalogo.csv"))
    hist = pd.read_csv(os.path.join(BASE, "ordenes_historial.csv"))
    hold = pd.read_csv(os.path.join(BASE, "ordenes_holdout.csv"))
    exp = pd.read_csv(os.path.join(BASE, "experimento.csv"))
    clientes = pd.read_csv(os.path.join(BASE, "clientes.csv"))
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probo algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values
    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# TODO 1 resuelto: tabla de similitudes producto x producto (diagonal en 0), calculada una sola vez
S_it = cosine_similarity(X.T)
np.fill_diagonal(S_it, 0)

# SVD (modelo libre del notebook): matriz completa de puntajes, calculada una vez
_svd = TruncatedSVD(n_components=8, random_state=42)
_Z = _svd.fit_transform(X)
S_SVD = _Z @ _svd.components_


def puntajes(metodo, x_vec, K, fila=None):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    if metodo == "Popularidad":
        return popularidad.astype(float)
    if metodo == "Filtrado por ítem":
        # TODO 2 resuelto: suma de las similitudes de cada producto con lo que el cliente ya compro
        return x_vec @ S_it
    if metodo == "SVD (8 factores)":
        return S_SVD[fila].astype(float)
    # TODO 3 resuelto: filtrado por cliente, K vecinos mas parecidos ponderados por similitud
    sims = cosine_similarity(x_vec[None, :], X)[0]
    sims[fila] = 0.0                                  # un cliente no es vecino de si mismo
    vec = np.argsort(-sims)[:K]
    return sims[vec] @ X[vec]


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


# --- Titulo ---
st.title("☕ Recomendador de productos")
st.caption("Ayuda a decidir que 3 productos mostrar en el carrusel Para ti de cada cliente, "
           "y a comparar que tan bien acierta cada recomendador antes de llevarlo a un experimento.")

# TODO 4 resuelto: controles interactivos en la barra lateral
with st.sidebar:
    st.header("Parámetros")
    metodo = st.selectbox("Método de recomendación",
                          ["Filtrado por ítem", "Filtrado por cliente", "SVD (8 factores)", "Popularidad"])
    K = st.slider("Número de vecinos K (solo filtrado por cliente)", 5, 200, 30, step=5)

tab1, tab2 = st.tabs(["Recomendar", "Comparar métodos"])

with tab1:
    cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()), value=14, step=1)
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    ya_compro = [productos[j] for j in np.where(x_vec > 0)[0]]
    st.write("Ya compró:", ya_compro if ya_compro else "nada en el historial (cliente sin historial)")

    sin_historial = x_vec.sum() == 0
    es_colaborativo = metodo in ("Filtrado por ítem", "Filtrado por cliente", "SVD (8 factores)")

    # TODO 5 resuelto: 3 recomendaciones con su puntaje y el "porque"
    if sin_historial and es_colaborativo:
        st.warning("Este cliente no tiene historial. Los métodos colaborativos no tienen de dónde partir, "
                   "así que se recomienda la popularidad como respaldo (arranque en frío).")
        scores = puntajes("Popularidad", x_vec, K, fila)
        metodo_usado = "Popularidad"
    else:
        scores = puntajes(metodo, x_vec, K, fila)
        metodo_usado = metodo

    rec = top3(scores, x_vec)
    st.subheader(f"Top 3 para el cliente {int(cid)} — {metodo_usado}")
    for rank, j in enumerate(rec, start=1):
        prod = productos[j]
        puntaje = scores[j]
        if metodo_usado == "Popularidad":
            porque = "es de lo más comprado por toda la base"
        elif metodo_usado == "Filtrado por ítem":
            comprados = [p for p in np.where(x_vec > 0)[0] if p != COL_AGUA]
            if comprados:
                mejor = max(comprados, key=lambda p: S_it[p, j])
                porque = f"porque compró {productos[mejor]} (similitud {S_it[mejor, j]:.2f})"
            else:
                porque = "por su perfil de compras"
        elif metodo_usado == "SVD (8 factores)":
            porque = "porque encaja con los factores de gusto que resumen sus compras"
        else:  # filtrado por cliente
            sims = cosine_similarity(x_vec[None, :], X)[0]
            sims[fila] = 0.0
            vec = np.argsort(-sims)[:K]
            cuantos = int(X[vec, j].sum())
            porque = f"porque {cuantos} de sus {K} vecinos más parecidos lo compraron"
        st.markdown(f"**{rank}. {prod}**  ·  puntaje {puntaje:.2f}  ·  {porque}")

with tab2:
    st.caption("Comparación sobre el grupo CONTROL (el que no vio el carrusel) y que probó algo nuevo "
               "en los 90 días del holdout. Un acierto aquí es una compra limpia, sin recomendación de por medio.")
    n_eval = st.slider("Cuántos clientes de control evaluar", 200, int(es_control.sum()), 2000, step=200)

    elegibles = NUEVO.sum(axis=1) > 0
    idx_control = np.where(es_control & elegibles)[0][:n_eval]

    def top_k_matriz(S, Kc=3):
        s = np.where(X > 0, -np.inf, S).astype(float)
        s[:, COL_AGUA] = -np.inf
        return np.argsort(-s, axis=1)[:, :Kc]

    # Matrices de puntajes de cada metodo (para toda la base)
    S_item_full = X @ S_it
    S_user_full = np.zeros_like(X)
    S_us = cosine_similarity(X); np.fill_diagonal(S_us, 0)
    for i in range(len(X)):
        v = np.argsort(-S_us[i])[:K]
        S_user_full[i] = S_us[i, v] @ X[v]
    S_pop_full = np.tile(popularidad, (len(X), 1))
    metodos_full = {"Popularidad": S_pop_full, "Filtrado por ítem": S_item_full,
                    "Filtrado por cliente": S_user_full, "SVD (8 factores)": S_SVD}

    filas_todas = np.arange(len(X))[:, None]
    comp = {}
    for nombre, S in metodos_full.items():
        topm = top_k_matriz(S)
        aciertos = NUEVO[filas_todas, topm].sum(axis=1) > 0
        hit = aciertos[idx_control].mean()
        cob = len(np.unique(topm))
        comp[nombre] = {"Hit rate@3 (control)": round(float(hit), 3), "Cobertura (de 28)": int(cob)}
    comp_df = pd.DataFrame(comp).T
    st.dataframe(comp_df, use_container_width=True)
    st.bar_chart(comp_df["Hit rate@3 (control)"])
    st.caption(f"Evaluado con K = {K} vecinos para el filtrado por cliente y {len(idx_control):,} clientes de control.")
