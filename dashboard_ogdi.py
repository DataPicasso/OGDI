import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px

# =================== #
# CONFIGURACIÓN GLOBAL
# =================== #
st.set_page_config(
    page_title="Dashboard OGDI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== #
# ESTILO PERSONALIZADO (CSS)
# =================== #
st.markdown("""
<style>
body, [data-testid="block-container"] {
    background: #f8fafd !important;
}

[data-testid="block-container"] {
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    padding-top: 1.5rem;
    padding-bottom: 0.5rem;
}

[data-testid="stMetric"] {
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 3px 12px rgba(44,62,80,0.06);
    padding: 20px 0 12px 0;
    margin-bottom: 12px;
    text-align: center;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 3px 12px rgba(44,62,80,0.07);
    padding: 1.3rem 1rem 1.1rem 1rem;
    margin-bottom: 18px;
}

hr {
    margin: 0.4em 0 0.8em 0;
    border-top: 1.5px solid #ebebeb;
}
</style>
""", unsafe_allow_html=True)

# =================== #
# SIDEBAR - Controles
# =================== #
with st.sidebar:
    st.title("Dashboard OGDI")
    st.write("**Observatorio de Gobierno Digital**")
    st.markdown("---")
    st.info("Usa los selectores del panel para comparar países y periodos. Las métricas y gráficos se actualizan automáticamente.")

# =================== #
# DATOS: Carga y preparación
# =================== #

# --- EGDI & E-Participation ---
df_egdi = pd.read_csv(r'C:\Users\pedro.figueroa\Desktop\Mother\AI\OGBI\EGOV_MERGED_2005_2024.csv')
years = [2005, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
egdi_cols = [f'E-Government Index {y}' for y in years]
ep_cols = [f'E-Participation Index {y}' for y in years]

# EGDI LONG
df_vis = df_egdi.melt(
    id_vars=['Country Name'],
    value_vars=egdi_cols,
    var_name='Year',
    value_name='EGDI'
)
df_vis['Year'] = df_vis['Year'].str.extract(r'(\d{4})').astype(int)
df_vis['EGDI'] = pd.to_numeric(df_vis['EGDI'], errors='coerce')
df_vis.dropna(subset=['EGDI'], inplace=True)
df_vis['Country Name'] = df_vis['Country Name'].str.strip()

# E-Participation LONG
df_ep = df_egdi.melt(
    id_vars=['Country Name'],
    value_vars=ep_cols,
    var_name='Year',
    value_name='E-Participation'
)
df_ep['Year'] = df_ep['Year'].str.extract(r'(\d{4})').astype(int)
df_ep['E-Participation'] = pd.to_numeric(df_ep['E-Participation'], errors='coerce')
df_ep.dropna(subset=['E-Participation'], inplace=True)
df_ep['Country Name'] = df_ep['Country Name'].str.strip()

# --- Internet Penetration ---
df_conn = pd.read_csv(r'C:\Users\pedro.figueroa\Desktop\Mother\AI\OGBI\internet_penetration_ogdi.csv')
yrs_conn = sorted(df_conn['Year'].unique())

# =================== #
# TÍTULO PRINCIPAL
# =================== #
st.markdown("""
#  Observatorio OGDI  
**Dashboard Interactivo Integrado**
""")
st.markdown("<hr>", unsafe_allow_html=True)

# =================== #
# PANEL PRINCIPAL (Layout horizontal tipo dashboard)
# =================== #
# 1. MÉTRICAS CLAVE | 2. EGDI Evolución y Predicción | 3. Top Países
col1, col2, col3 = st.columns([1.6, 3.5, 1.7], gap="medium")

# ========= #
# COL 1: KPIs
# ========= #
with col1:
    st.markdown("### 🏆 Métricas clave")
    # Mayor EGDI actual
    year_max = df_vis['Year'].max()
    top_now = df_vis[df_vis['Year']==year_max].sort_values('EGDI', ascending=False).iloc[0]
    st.metric(label=f"Máximo EGDI ({year_max})", value=f"{top_now['Country Name']}", delta=f"{top_now['EGDI']:.3f}")

    # Promedio regional (personalizable)
    promedio_actual = df_vis[df_vis['Year']==year_max]['EGDI'].mean()
    st.metric(label=f"Promedio Global ({year_max})", value=f"{promedio_actual:.3f}")

    # EGDI RD actual (si existe)
    rd_actual = df_vis[(df_vis['Country Name']=="Dominican Republic") & (df_vis['Year']==year_max)]
    if not rd_actual.empty:
        rd_val = rd_actual.iloc[0]['EGDI']
        st.metric(label="RD EGDI actual", value=f"{rd_val:.3f}")

    st.markdown("### 📬 Participación Electrónica")
    ep_max = df_ep[df_ep['Year']==year_max].sort_values('E-Participation', ascending=False).iloc[0]
    st.metric(label=f"Máxima E-Part ({year_max})", value=ep_max['Country Name'], delta=f"{ep_max['E-Participation']:.3f}")

# ========= #
# COL 2: Gráficos EGDI + ML
# ========= #
with col2:
    st.markdown("#### Evolución EGDI + Predicción ML")
    # Selectores principales
    paises = sorted(df_vis['Country Name'].unique())
    seleccionados = st.multiselect(
        "🌎 Países a comparar",
        paises,
        default=["Dominican Republic"],
        key="egdi_select"
    )
    min_y, max_y = int(df_vis['Year'].min()), int(df_vis['Year'].max())
    rango_anos = st.slider(
        "📅 Rango de años",
        min_y, max_y,
        (min_y, max_y),
        step=1,
        key="egdi_slider"
    )

    # --- Gráfico EGDI histórico ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pais in seleccionados:
        d = df_vis[(df_vis['Country Name'] == pais) &
                   (df_vis['Year'].between(*rango_anos))]
        ax.plot(d['Year'], d['EGDI'], 'o-', lw=2, label=pais)
    ax.set_title("Evolución del EGDI")
    ax.set_xlabel("Año")
    ax.set_ylabel("EGDI")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # --- Predicción ML ---
    pais_pred = st.selectbox("🤖 Predicción ML para:", paises,
                             index=paises.index("Dominican Republic"),
                             key="ml_pred_select")
    d_pred = df_vis[df_vis['Country Name'] == pais_pred].sort_values('Year')
    if len(d_pred) > 1:
        X = d_pred['Year'].values.reshape(-1, 1)
        y = d_pred['EGDI'].values
        model = LinearRegression().fit(X, y)
        futuros = np.array([2025, 2026]).reshape(-1, 1)
        preds = model.predict(futuros)

        fig2, ax2 = plt.subplots(figsize=(5, 2.5))
        ax2.plot(d_pred['Year'], d_pred['EGDI'], 'o-', label="Histórico")
        ax2.plot([2025, 2026], preds, 'x--', color='red', label="Predicción")
        ax2.set_title(f"Predicción EGDI para {pais_pred}")
        ax2.set_xlabel("Año")
        ax2.set_ylabel("EGDI")
        ax2.grid(alpha=0.25)
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)

        trend = "alcista" if model.coef_[0] > 0 else "bajista"
        st.info(f"**Tendencia {trend}** • 2026 ≈ {preds[-1]:.3f} (actual {y[-1]:.3f}) • Δ anual {model.coef_[0]:.4f}")
    else:
        st.warning("❗ Datos insuficientes para predicción")

# ========= #
# COL 3: Tabla top países
# ========= #
with col3:
    st.markdown("### 🥇 Top Países EGDI")
    df_top = df_vis[df_vis['Year']==year_max].sort_values('EGDI', ascending=False).head(8)[['Country Name', 'EGDI']]
    st.dataframe(df_top, hide_index=True, use_container_width=True)
    st.markdown("### ℹ️ Info")
    with st.expander("Sobre estos datos"):
        st.write("""
        - **EGDI**: E-Government Development Index (ONU)
        - **Predicción ML**: Regresión lineal, tendencia anual calculada automáticamente.
        - Los datos provienen de informes ONU (2024).
        """)

st.markdown("<hr>", unsafe_allow_html=True)

# =================== #
# PANEL 2: E-Participation + Anomalías
# =================== #
colA, colB = st.columns([2.7, 1.3], gap="medium")
with colA:
    st.markdown("#### Evolución E-Participation y Alertas")
    sel_ep = st.multiselect(
        "🌐 Países a comparar (E-Part)",
        sorted(df_ep['Country Name'].unique()),
        default=["Dominican Republic", "Chile", "Uruguay"],
        key="ep_select"
    )
    rango_ep = st.slider(
        "📅 Rango de años (E-Part)",
        int(df_ep['Year'].min()), int(df_ep['Year'].max()),
        (int(df_ep['Year'].min()), int(df_ep['Year'].max())),
        step=1,
        key="ep_slider"
    )

    def detect_anomalies(arr):
        dif = np.diff(arr)
        if len(dif) < 2:
            return np.zeros(len(arr), dtype=bool)
        z = (dif - dif.mean()) / (dif.std() or 1)
        flags = np.insert(np.abs(z) > 2, 0, False)
        return flags

    fig3, ax3 = plt.subplots(figsize=(8, 4.2))
    for p in sel_ep:
        d = df_ep[(df_ep['Country Name']==p) &
                  (df_ep['Year'].between(*rango_ep))].sort_values('Year')
        ax3.plot(d['Year'], d['E-Participation'], 'o-', lw=2, label=p)
        flags = detect_anomalies(d['E-Participation'].values)
        if flags.any():
            ax3.scatter(d['Year'][flags], d['E-Participation'][flags],
                        color='red', marker='x', s=100, label=f"Alerta {p}")
    ax3.set_title("E-Participation Index con Alertas")
    ax3.set_xlabel("Año")
    ax3.set_ylabel("Index")
    ax3.grid(alpha=0.2)
    ax3.legend()
    st.pyplot(fig3, use_container_width=True)

with colB:
    st.markdown("#### 🔔 Insights automáticos")
    for p in sel_ep:
        d = df_ep[(df_ep['Country Name']==p) &
                  (df_ep['Year'].between(*rango_ep))].sort_values('Year')
        flags = detect_anomalies(d['E-Participation'].values)
        if flags.any():
            for i, f in enumerate(flags):
                if f:
                    yr = d['Year'].iloc[i]
                    val = d['E-Participation'].iloc[i]
                    st.error(f"{p} en {yr}: cambio atípico ({val:.3f})")
        else:
            st.success(f"{p}: sin retrocesos bruscos")

st.markdown("<hr>", unsafe_allow_html=True)

# =================== #
# PANEL 3: Clusters Internet Penetration
# =================== #

# --- Calcula solo los años con suficiente data ---
years_with_data = (
    df_conn
    .dropna(subset=['Internet Penetration'])
    .groupby('Year')['Country Name']
    .count()
    .reset_index()
)
# Aquí definimos mínimo 4 países, puedes ajustar si quieres.
years_with_data = years_with_data[years_with_data['Country Name'] >= 20]['Year'].tolist()

if len(years_with_data) == 0:
    st.error("No hay años con suficiente data para mostrar el mapa de clusters.")
else:
    # El valor por defecto será el último año con data suficiente.
    default_year = max(years_with_data)

    colC, colD = st.columns([2.5, 1.5], gap="medium")
    with colC:
        st.markdown("#### 🌐 Mapa de Clusters de Penetración de Internet")
        year_sel = st.slider(
            "📅 Año a analizar",
            min(years_with_data),
            max(years_with_data),
            value=default_year,
            key="conn_slider"
        )
        df_y = df_conn[(df_conn['Year']==year_sel) & (~df_conn['Internet Penetration'].isna())]
        X = df_y[['Internet Penetration']].values
        km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
        df_y['Cluster'] = km.labels_
        means = df_y.groupby('Cluster')['Internet Penetration'].mean().sort_values()
        labels = ['Muy baja', 'Media-baja', 'Media-alta', 'Alta']
        mapping = {c: labels[i] for i, c in enumerate(means.index)}
        df_y['Nivel'] = df_y['Cluster'].map(mapping)

        # --- Mapa Plotly ---
        fig4 = px.choropleth(
            df_y,
            locations="Country Name",
            locationmode="country names",
            color="Nivel",
            category_orders={"Nivel": labels},
            hover_name="Country Name",
            hover_data={"Internet Penetration":":.2f"},
            title=f"Penetración de Internet en {year_sel}",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig4.update_geos(
            visible=False, resolution=50, showcountries=True, countrycolor="LightGray",
            lataxis_range=[10,25], lonaxis_range=[-90,-60]
        )
        fig4.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig4, use_container_width=True)

    with colD:
        st.markdown("#### 📋 Resumen de niveles")
        cols = st.columns(2)
        for i, lvl in enumerate(labels):
            subset = df_y[df_y['Nivel']==lvl]
            texto = f"**{lvl}**: {len(subset)} países — Promedio {subset['Internet Penetration'].mean():.1f}%"
            ejemplos = ", ".join(subset['Country Name'].head(6).tolist())
            cols[i%2].markdown(texto)
            cols[i%2].caption(f"Ejemplos: {ejemplos}…")

        # --- Enfoque RD ---
        rd = df_y[df_y['Country Name']=="Dominican Republic"]
        if not rd.empty:
            lvl = rd['Nivel'].iloc[0]
            val = rd['Internet Penetration'].iloc[0]
            st.success(f"🇩🇴 República Dominicana: _{lvl}_ ({val:.1f}%)")
        else:
            st.warning("❗ Sin datos para RD")

    st.markdown("<hr>", unsafe_allow_html=True)


st.caption("Dashboard OGDI • Datos: ONU & Banco Mundial ")
