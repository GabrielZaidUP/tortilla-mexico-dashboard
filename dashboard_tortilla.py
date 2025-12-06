"""
============================================================================
DASHBOARD STREAMLIT - PRECIOS DE TORTILLA EN MÉXICO
============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
from sklearn.linear_model import LinearRegression




from analisis_funciones import (
    load_all_data, calcular_analisis_presidencial, 
    calcular_impacto_eventos, calcular_estadisticas_estatales,
    calcular_proyecciones, export_results_to_csv, export_results_to_json,
    PRESIDENTES, EVENTOS_CRITICOS, COLORS, OUTPUT_DIR,
    CONSUMO_PROMEDIO_PERSONA, PERSONAS_HOGAR, DIAS_MES_PROMEDIO,
    to_datetime, to_datetime_array,load_mexico_geojson, prepare_map_data,prepare_temporal_map_data,
    get_presidente_actual
)

st.set_page_config(
    page_title="🌮 Análisis Tortilla México",
    page_icon="🌮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #FFE66D 0%, #FF6B6B 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B6B;
    }
    .conclusion-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4ECDC4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_todos_los_datos():
    return load_all_data()

@st.cache_data
def cargar_geojson():
    """Carga GeoJSON de México con caché"""
    return load_mexico_geojson()

with st.spinner('📄 Cargando y procesando datos...'):
    df_tortilla, df_maiz, df_salario, df_pobreza, df_itlp_nacional, df_master, HAS_CITIES = cargar_todos_los_datos()
    df_pres = calcular_analisis_presidencial(df_master)
    df_eventos = calcular_impacto_eventos(df_master)
    state_stats = calcular_estadisticas_estatales(df_tortilla)
    proyecciones = calcular_proyecciones(df_master)
    
    # NUEVO: Cargar datos geográficos
    mexico_geojson = cargar_geojson()
    if mexico_geojson:
        state_map_data = prepare_map_data(df_tortilla)
    else:
        state_map_data = None

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📊 **{len(df_master):,} meses** de datos históricos")
with col2:
    st.info(f"📅 **{df_master['Date'].min().year} - {df_master['Date'].max().year}** Periodo analizado")
with col3:
    st.info(f"🗺️ **{len(state_stats)} estados** monitoreados")

st.markdown("---")

st.sidebar.title("🎛️ Configuración")
st.sidebar.markdown("### 📅 Rango de Fechas")
fecha_min = df_master['Date'].min().date()
fecha_max = df_master['Date'].max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    fecha_inicio = st.date_input("Desde", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
with col2:
    fecha_fin = st.date_input("Hasta", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

st.sidebar.markdown("### 🏛️ Presidentes")
presidentes_disponibles = list(PRESIDENTES.keys())
presidentes_seleccionados = st.sidebar.multiselect(
    "Administraciones",
    options=presidentes_disponibles,
    default=presidentes_disponibles
)

st.sidebar.markdown("### ⚡ Eventos Críticos")
mostrar_eventos = st.sidebar.checkbox("Mostrar eventos", value=True)
if mostrar_eventos:
    tipos_eventos = list(set([e['tipo'] for e in EVENTOS_CRITICOS]))
    tipos_seleccionados = st.sidebar.multiselect(
        "Tipos",
        options=tipos_eventos,
        default=tipos_eventos,
        format_func=lambda x: {'economico': '💰 Económicos', 'sanitario': '🏥 Sanitarios', 'politico': '🏛️ Políticos'}.get(x, x)
    )
else:
    tipos_seleccionados = []

st.sidebar.markdown("### 🎨 Visualización")
mostrar_bandas = st.sidebar.checkbox("Bandas presidenciales", value=True)
mostrar_tendencia = st.sidebar.checkbox("Líneas de tendencia", value=False)

df_filtered = df_master[
    (df_master['Date'].dt.date >= fecha_inicio) &
    (df_master['Date'].dt.date <= fecha_fin)
].copy()

if presidentes_seleccionados:
    df_filtered = df_filtered[df_filtered['Presidente'].isin(presidentes_seleccionados)]

st.markdown("## 📊 Indicadores Clave")

latest = df_master.iloc[-1]
previous_month = df_master.iloc[-2] if len(df_master) > 1 else latest
previous_year = df_master.iloc[-13] if len(df_master) >= 13 else df_master.iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    cambio_mensual = ((latest['Precio_Tortilla'] / previous_month['Precio_Tortilla']) - 1) * 100
    st.metric(
        "💵 Precio Actual",
        f"${latest['Precio_Tortilla']:.2f}",
        f"{cambio_mensual:+.2f}% mes",
        delta_color="inverse"
    )

with col2:
    cambio_anual = ((latest['Precio_Tortilla'] / previous_year['Precio_Tortilla']) - 1) * 100
    st.metric(
        "📈 Cambio Anual",
        f"{cambio_anual:+.1f}%",
        help="Variación respecto hace 12 meses"
    )

with col3:
    st.metric(
        "🎯 Accesibilidad",
        f"{latest['Indice_Accesibilidad']:.1f}/100",
        f"{latest['Indice_Accesibilidad'] - 100:+.1f} pts vs 2007"
    )

with col4:
    st.metric(
        "🛒 Kg/Salario",
        f"{latest['Kg_Tortilla_x_Salario']:.0f} kg",
        help="Kilogramos que se pueden comprar con un salario mínimo mensual"
    )

with col5:
    corr_maiz = df_master[['Precio_Tortilla', 'INPP_Maiz']].corr().iloc[0, 1]
    st.metric(
        "🌽 Corr. Maíz",
        f"{corr_maiz:.3f}",
        help="Correlación con el Índice de Precios del Maíz"
    )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9,tab10 = st.tabs([
    "📈 Timeline Principal",
    "🏛️ Análisis Presidencial", 
    "⚡ Impacto de Eventos",
    "🗺️ Análisis Regional",
    "💰 Poder Adquisitivo",
    "🌽 Correlaciones",
    "🔮 Proyecciones",
    "🏙️ Análisis por Ciudades",  
    "📊 Disparidad Regional",  
    "📄 Conclusiones"
])

# TAB 1
with tab1:
    st.markdown("## 📈 Evolución Histórica del Precio de la Tortilla")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Precio_Tortilla'],
        mode='lines',
        name='Precio Tortilla',
        line=dict(color=COLORS['primary'], width=3),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>$%{y:.2f}/kg<extra></extra>'
    ))
    
    if mostrar_bandas:
        shapes = []
        annotations = []
        for nombre, info in PRESIDENTES.items():
            if nombre in presidentes_seleccionados:
                inicio = pd.Timestamp(info['inicio'])
                fin = pd.Timestamp(info['fin'])
                
                if fin >= df_filtered['Date'].min() and inicio <= df_filtered['Date'].max():
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=inicio,
                        x1=fin,
                        y0=0,
                        y1=1,
                        fillcolor=info['color'],
                        opacity=0.1,
                        layer="below",
                        line_width=0
                    ))
                    
                    annotations.append(dict(
                        x=inicio,
                        y=1,
                        xref="x",
                        yref="paper",
                        text=f"{nombre}<br>({info['partido']})",
                        showarrow=False,
                        xanchor="left",
                        yanchor="top",
                        font=dict(size=10)
                    ))
        
        fig.update_layout(shapes=shapes, annotations=annotations)
    
    if mostrar_eventos:
        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        annotations = list(fig.layout.annotations) if fig.layout.annotations else []
        
        for evento in EVENTOS_CRITICOS:
            if evento['tipo'] in tipos_seleccionados:
                fecha_evento = pd.Timestamp(evento['fecha'])
                
                if fecha_evento >= df_filtered['Date'].min() and fecha_evento <= df_filtered['Date'].max():
                    color_evento = {
                        'economico': COLORS['crisis'], 
                        'sanitario': COLORS['sanitario'], 
                        'politico': COLORS['politico']
                    }.get(evento['tipo'], '#666')
                    
                    shapes.append(dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=fecha_evento,
                        x1=fecha_evento,
                        y0=0,
                        y1=1,
                        line=dict(
                            color=color_evento,
                            width=2,
                            dash="dash"
                        )
                    ))
                    
                    annotations.append(dict(
                        x=fecha_evento,
                        y=1,
                        xref="x",
                        yref="paper",
                        text=evento['nombre'],
                        showarrow=False,
                        textangle=-90,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=9, color=color_evento)
                    ))
        
        fig.update_layout(shapes=shapes, annotations=annotations)
    
    if mostrar_tendencia:
        X = np.arange(len(df_filtered)).reshape(-1, 1)
        y = df_filtered['Precio_Tortilla'].values
        model = LinearRegression().fit(X, y)
        
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=model.predict(X),
            mode='lines',
            name='Tendencia',
            line=dict(color=COLORS['purple'], width=2, dash='dash'),
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Precio de la Tortilla a lo Largo del Tiempo",
        xaxis_title="Fecha",
        yaxis_title="Precio (MXN/kg)",
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Estadísticas del Período Seleccionado")
    col1, col2, col3, col4 = st.columns(4)
    
    stats = df_filtered['Precio_Tortilla'].describe()
    with col1:
        st.metric("📊 Promedio", f"${stats['mean']:.2f}")
    with col2:
        st.metric("📉 Mínimo", f"${stats['min']:.2f}")
    with col3:
        st.metric("📈 Máximo", f"${stats['max']:.2f}")
    with col4:
        st.metric("📏 Rango", f"${stats['max'] - stats['min']:.2f}")

# TAB 2
with tab2:
    st.markdown("## 🏛️ Análisis Comparativo por Administración Presidencial")
    
    df_pres_filtered = df_pres[df_pres['Presidente'].isin(presidentes_seleccionados)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_inc = px.bar(
            df_pres_filtered,
            x='Presidente',
            y='Incremento_Porcentual',
            color='Partido',
            color_discrete_map={'PAN': COLORS['calderon'], 'PRI': COLORS['pena'], 'MORENA': COLORS['amlo']},
            title='Incremento Total por Administración (%)',
            text='Incremento_Porcentual'
        )
        fig_inc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_inc.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_inc, use_container_width=True)
    
    with col2:
        fig_cagr = px.bar(
            df_pres_filtered,
            x='Presidente',
            y='CAGR',
            color='Partido',
            color_discrete_map={'PAN': COLORS['calderon'], 'PRI': COLORS['pena'], 'MORENA': COLORS['amlo']},
            title='CAGR Anualizado (%)',
            text='CAGR'
        )
        fig_cagr.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_cagr.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cagr, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_vol = px.bar(
            df_pres_filtered,
            x='Presidente',
            y='Volatilidad',
            color='Partido',
            color_discrete_map={'PAN': COLORS['calderon'], 'PRI': COLORS['pena'], 'MORENA': COLORS['amlo']},
            title='Volatilidad de Precios (%)',
            text='Volatilidad'
        )
        fig_vol.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_vol.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col4:
        fig_prom = px.bar(
            df_pres_filtered,
            x='Presidente',
            y='Promedio',
            color='Partido',
            color_discrete_map={'PAN': COLORS['calderon'], 'PRI': COLORS['pena'], 'MORENA': COLORS['amlo']},
            title='Precio Promedio (MXN/kg)',
            text='Promedio'
        )
        fig_prom.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_prom.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_prom, use_container_width=True)
    
    st.markdown("### 📋 Tabla Resumen")
    st.dataframe(
        df_pres_filtered[['Presidente', 'Partido', 'Precio_Inicial', 'Precio_Final', 
                         'Incremento_Porcentual', 'CAGR', 'Volatilidad', 'Promedio']].style.format({
            'Precio_Inicial': '${:.2f}',
            'Precio_Final': '${:.2f}',
            'Incremento_Porcentual': '{:+.2f}%',
            'CAGR': '{:.2f}%',
            'Volatilidad': '{:.2f}%',
            'Promedio': '${:.2f}'
        }),
        use_container_width=True
    )
    
    if st.button("💾 Exportar Análisis Presidencial", key="export_pres"):
        path = export_results_to_csv(df_pres_filtered, "analisis_presidencial.csv")
        st.success(f"✅ Exportado: {path}")
        
        csv = df_pres_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Descargar CSV",
            data=csv,
            file_name="analisis_presidencial.csv",
            mime="text/csv"
        )

# TAB 3
with tab3:
    st.markdown("## ⚡ Análisis de Impacto de Eventos Críticos")
    
    df_eventos_filtered = df_eventos[df_eventos['Tipo'].isin(tipos_seleccionados)] if tipos_seleccionados else df_eventos
    
    fig_eventos = px.bar(
        df_eventos_filtered.sort_values('Cambio_Porcentual'),
        x='Cambio_Porcentual',
        y='Evento',
        orientation='h',
        color='Tipo',
        color_discrete_map={'economico': COLORS['crisis'], 
                           'sanitario': COLORS['sanitario'], 
                           'politico': COLORS['politico']},
        title='Impacto de Eventos en el Precio (% cambio 6 meses antes/después)',
        text='Cambio_Porcentual',
        hover_data=['Fecha', 'Precio_Antes', 'Precio_Despues']
    )
    fig_eventos.update_traces(texttemplate='%{text:+.1f}%', textposition='outside')
    fig_eventos.update_layout(height=500)
    st.plotly_chart(fig_eventos, use_container_width=True)
    
    st.markdown("### 📋 Detalle de Eventos")
    st.dataframe(
        df_eventos_filtered.style.format({
            'Precio_Antes': '${:.2f}',
            'Precio_Despues': '${:.2f}',
            'Cambio_Absoluto': '${:+.2f}',
            'Cambio_Porcentual': '{:+.2f}%'
        }),
        use_container_width=True
    )
    
    if st.button("💾 Exportar Análisis de Eventos", key="export_eventos"):
        path = export_results_to_csv(df_eventos_filtered, "impacto_eventos.csv")
        st.success(f"✅ Exportado: {path}")

# TAB 4
with tab4:
    st.markdown("## 🗺️ Análisis por Estado")
    
    # ========================================================================
    # SECCIÓN: MAPA INTERACTIVO
    # ========================================================================
    if mexico_geojson and state_map_data is not None:
        st.markdown("### 🌎 Mapa Interactivo de Precios")
        
        # Selector de métrica para colorear el mapa
        col1, col2 = st.columns([3, 1])
        
        with col1:
            metrica_mapa = st.selectbox(
                "Selecciona métrica a visualizar:",
                options=['Precio_Promedio', 'Precio_Min', 'Precio_Max', 'Std', 'CV', 'Rango'],
                format_func=lambda x: {
                    'Precio_Promedio': '💵 Precio Promedio Histórico',
                    'Precio_Min': '📉 Precio Mínimo Histórico',
                    'Precio_Max': '📈 Precio Máximo Histórico',
                    'Std': '📊 Desviación Estándar',
                    'CV': '📐 Coeficiente de Variación (%)',
                    'Rango': '📏 Rango de Precios'
                }.get(x, x)
            )
        
        with col2:
            colorscale = st.selectbox(
                "Escala de color:",
                options=['RdYlGn_r', 'Viridis', 'Plasma', 'Reds', 'Blues'],
                index=0
            )
        
        # Crear mapa choropleth
        fig_map = px.choropleth(
            state_map_data,
            geojson=mexico_geojson,
            locations='State',
            featureidkey="properties.name",
            color=metrica_mapa,
            color_continuous_scale=colorscale,
            hover_name='State',
            hover_data={
                'State': False,  # No mostrar duplicado
                'Precio_Promedio': ':.2f',
                'Precio_Min': ':.2f',
                'Precio_Max': ':.2f',
                'Std': ':.2f',
                'CV': ':.2f',
                'Rango': ':.2f',
                'Registros': ':,',
                'Año_Inicio': True,
                'Año_Fin': True
            },
            labels={
                'Precio_Promedio': 'Precio Prom. ($)',
                'Precio_Min': 'Precio Mín. ($)',
                'Precio_Max': 'Precio Máx. ($)',
                'Std': 'Desv. Std ($)',
                'CV': 'CV (%)',
                'Rango': 'Rango ($)',
                'Registros': 'Datos',
                'Año_Inicio': 'Desde',
                'Año_Fin': 'Hasta'
            },
            title=f"Mapa de México: {metrica_mapa.replace('_', ' ')}"
        )
        
        # Configurar el layout del mapa
        fig_map.update_geos(
            fitbounds="locations",
            visible=False,
            showcountries=False,
            showcoastlines=False,
            showland=False,
            showlakes=False
        )
        
        fig_map.update_layout(
            height=700,
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title=metrica_mapa.replace('_', ' '),
                thicknessmode="pixels",
                thickness=15,
                lenmode="pixels",
                len=400,
                yanchor="middle",
                y=0.5
            )
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Información adicional del mapa
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_state = state_map_data.loc[state_map_data[metrica_mapa].idxmax()]
            st.metric(
                "🔴 Estado Máximo",
                max_state['State'],
                f"${max_state[metrica_mapa]:.2f}" if 'Precio' in metrica_mapa else f"{max_state[metrica_mapa]:.2f}"
            )
        
        with col2:
            min_state = state_map_data.loc[state_map_data[metrica_mapa].idxmin()]
            st.metric(
                "🟢 Estado Mínimo",
                min_state['State'],
                f"${min_state[metrica_mapa]:.2f}" if 'Precio' in metrica_mapa else f"{min_state[metrica_mapa]:.2f}"
            )
        
        with col3:
            brecha = max_state[metrica_mapa] - min_state[metrica_mapa]
            st.metric(
                "📊 Brecha",
                f"${brecha:.2f}" if 'Precio' in metrica_mapa else f"{brecha:.2f}",
                f"{(brecha/min_state[metrica_mapa]*100):.1f}%"
            )
        
        st.markdown("---")
        
        # ====================================================================
        # SECCIÓN: MAPA ANIMADO POR AÑO
        # ====================================================================
        st.markdown("### 🎬 Evolución Temporal por Año")
        
        years_available = sorted(df_tortilla['Year'].unique())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_year = st.select_slider(
                "Selecciona año:",
                options=years_available,
                value=years_available[-1]
            )
        
        with col2:
            st.metric("📅 Año Seleccionado", selected_year)
        
        # Preparar datos para el año seleccionado
        year_data = prepare_temporal_map_data(df_tortilla, selected_year)
        
        # Obtener presidente del año
        presidente, partido = get_presidente_actual(f"{selected_year}-06-01")
        
        # Crear mapa del año
        fig_year = px.choropleth(
            year_data,
            geojson=mexico_geojson,
            locations='State',
            featureidkey="properties.name",
            color='Precio',
            color_continuous_scale='RdYlGn_r',
            hover_name='State',
            hover_data={
                'State': False,
                'Precio': ':.2f',
                'Año': True
            },
            labels={
                'Precio': 'Precio (MXN/kg)',
                'Año': 'Año'
            },
            title=f"Precio de la Tortilla por Estado - {selected_year}"
        )
        
        # Agregar anotación del presidente
        if presidente:
            color_pres = PRESIDENTES[presidente]['color']
            fig_year.add_annotation(
                text=f"<b>Presidente: {presidente}</b><br>({partido})",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                showarrow=False,
                bgcolor=color_pres,
                opacity=0.7,
                bordercolor="black",
                borderwidth=2,
                borderpad=8,
                font=dict(size=14, color="white", family="Arial Black")
            )
        
        fig_year.update_geos(
            fitbounds="locations",
            visible=False
        )
        
        fig_year.update_layout(
            height=650,
            margin={"r": 0, "t": 80, "l": 0, "b": 0}
        )
        
        st.plotly_chart(fig_year, use_container_width=True)
        
        # Estadísticas del año
        st.markdown(f"#### 📊 Estadísticas {selected_year}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        year_stats = year_data['Precio'].describe()
        
        with col1:
            st.metric("💵 Promedio", f"${year_stats['mean']:.2f}")
        with col2:
            st.metric("📉 Mínimo", f"${year_stats['min']:.2f}")
        with col3:
            st.metric("📈 Máximo", f"${year_stats['max']:.2f}")
        with col4:
            st.metric("📏 Rango", f"${year_stats['max'] - year_stats['min']:.2f}")
        
        # Comparación interanual si no es el primer año
        if selected_year > years_available[0]:
            st.markdown("#### 📈 Comparación Interanual")
            
            year_anterior = selected_year - 1
            year_data_anterior = prepare_temporal_map_data(df_tortilla, year_anterior)
            
            # Merge para comparar
            comparison = year_data.merge(
                year_data_anterior,
                on='State',
                suffixes=('_actual', '_anterior')
            )
            comparison['Cambio_Absoluto'] = comparison['Precio_actual'] - comparison['Precio_anterior']
            comparison['Cambio_Porcentual'] = (comparison['Cambio_Absoluto'] / comparison['Precio_anterior']) * 100
            
            # Mapa de cambio
            fig_change = px.choropleth(
                comparison,
                geojson=mexico_geojson,
                locations='State',
                featureidkey="properties.name",
                color='Cambio_Porcentual',
                color_continuous_scale='RdYlGn_r',
                color_continuous_midpoint=0,
                hover_name='State',
                hover_data={
                    'State': False,
                    'Precio_actual': ':.2f',
                    'Precio_anterior': ':.2f',
                    'Cambio_Absoluto': ':+.2f',
                    'Cambio_Porcentual': ':+.2f'
                },
                labels={
                    'Precio_actual': f'Precio {selected_year} ($)',
                    'Precio_anterior': f'Precio {year_anterior} ($)',
                    'Cambio_Absoluto': 'Cambio ($)',
                    'Cambio_Porcentual': 'Cambio (%)'
                },
                title=f"Cambio Porcentual: {year_anterior} → {selected_year}"
            )
            
            fig_change.update_geos(
                fitbounds="locations",
                visible=False
            )
            
            fig_change.update_layout(
                height=600,
                margin={"r": 0, "t": 50, "l": 0, "b": 0}
            )
            
            st.plotly_chart(fig_change, use_container_width=True)
            
            # Top 5 mayores incrementos/decrementos
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📈 Top 5 Mayores Incrementos")
                top_incrementos = comparison.nlargest(5, 'Cambio_Porcentual')[
                    ['State', 'Cambio_Porcentual', 'Cambio_Absoluto']
                ]
                st.dataframe(
                    top_incrementos.style.format({
                        'Cambio_Porcentual': '{:+.2f}%',
                        'Cambio_Absoluto': '${:+.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.markdown("##### 📉 Top 5 Menores Cambios")
                top_decrementos = comparison.nsmallest(5, 'Cambio_Porcentual')[
                    ['State', 'Cambio_Porcentual', 'Cambio_Absoluto']
                ]
                st.dataframe(
                    top_decrementos.style.format({
                        'Cambio_Porcentual': '{:+.2f}%',
                        'Cambio_Absoluto': '${:+.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        
        st.markdown("---")
    
    else:
        st.warning("⚠️ Datos geográficos no disponibles. Asegúrate de tener el archivo `mexicoHigh.json` en `./data/`")
        st.info("""
        **Cómo obtener el archivo GeoJSON:**
        1. Descarga de: https://github.com/angelnmara/geojson/blob/master/mexicoHigh.json
        2. Colócalo en la carpeta `./data/` de tu proyecto
        """)
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Top 10 Estados Más Caros")
        top10 = state_stats.head(10).reset_index()
        
        fig_top = px.bar(
            top10,
            y='State',
            x='mean',
            orientation='h',
            color='mean',
            color_continuous_scale='Reds',
            text='mean',
            labels={'mean': 'Precio Promedio (MXN/kg)'}
        )
        fig_top.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_top.update_layout(height=450, showlegend=False, xaxis_title="Precio Promedio")
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        st.markdown("### 🟢 Top 10 Estados Más Baratos")
        bottom10 = state_stats.tail(10).sort_values('mean').reset_index()
        
        fig_bottom = px.bar(
            bottom10,
            y='State',
            x='mean',
            orientation='h',
            color='mean',
            color_continuous_scale='Greens',
            text='mean',
            labels={'mean': 'Precio Promedio (MXN/kg)'}
        )
        fig_bottom.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_bottom.update_layout(height=450, showlegend=False, xaxis_title="Precio Promedio")
        st.plotly_chart(fig_bottom, use_container_width=True)
    
    st.markdown("### 📊 Análisis de Volatilidad")
    col3, col4 = st.columns(2)
    
    with col3:
        vol_abs = state_stats.nlargest(15, 'std').reset_index()
        fig_vol_abs = px.bar(
            vol_abs,
            y='State',
            x='std',
            orientation='h',
            title='Top 15 Estados con Mayor Volatilidad (Desv. Std)',
            text='std',
            color='std',
            color_continuous_scale='OrRd'
        )
        fig_vol_abs.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_vol_abs.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_vol_abs, use_container_width=True)
    
    with col4:
        vol_cv = state_stats.nlargest(15, 'cv').reset_index()
        fig_vol_cv = px.bar(
            vol_cv,
            y='State',
            x='cv',
            orientation='h',
            title='Top 15 Estados con Mayor CV (%)',
            text='cv',
            color='cv',
            color_continuous_scale='YlOrRd'
        )
        fig_vol_cv.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_vol_cv.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_vol_cv, use_container_width=True)
    
    st.markdown("### 📋 Estadísticas Completas por Estado")
    st.dataframe(
        state_stats.style.format({
            'mean': '${:.2f}',
            'median': '${:.2f}',
            'std': '${:.2f}',
            'min': '${:.2f}',
            'max': '${:.2f}',
            'range': '${:.2f}',
            'cv': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    if st.button("💾 Exportar Análisis Estatal", key="export_states"):
        path = export_results_to_csv(state_stats.reset_index(), "estadisticas_estatales.csv")
        st.success(f"✅ Exportado: {path}")
    
    st.markdown("---")
    st.markdown("### 💾 Exportar Datos Geográficos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Exportar Estadísticas por Estado", key="export_geo_stats"):
            if state_map_data is not None:
                path = export_results_to_csv(state_map_data, "estadisticas_estados_completas.csv")
                st.success(f"✅ Exportado: {path}")
                
                csv = state_map_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar CSV",
                    data=csv,
                    file_name="estadisticas_estados_completas.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📥 Exportar Datos del Año Seleccionado", key="export_year_data"):
            if mexico_geojson:
                year_export = prepare_temporal_map_data(df_tortilla, selected_year)
                path = export_results_to_csv(year_export, f"precios_estados_{selected_year}.csv")
                st.success(f"✅ Exportado: {path}")
                
                csv_year = year_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar CSV",
                    data=csv_year,
                    file_name=f"precios_estados_{selected_year}.csv",
                    mime="text/csv"
                )

# ============================================================================
# 7. BONUS: AGREGAR COMPARACIÓN MULTI-ESTADO EN EL MAPA
# (Opcional - Agregar después del mapa animado)
# ============================================================================

        st.markdown("---")
        st.markdown("### 🔍 Comparador de Estados")
        
        estados_comparar = st.multiselect(
            "Selecciona estados para comparar en detalle:",
            options=sorted(state_map_data['State'].tolist()),
            default=state_map_data.nlargest(3, 'Precio_Promedio')['State'].tolist(),
            max_selections=5
        )
        
        if estados_comparar:
            # Datos filtrados
            estados_comparison = state_map_data[state_map_data['State'].isin(estados_comparar)]
            
            # Gráfico de barras comparativo
            fig_compare = go.Figure()
            
            metrics = ['Precio_Promedio', 'Precio_Min', 'Precio_Max', 'CV']
            colors_compare = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#9B59B6']
            
            for i, metric in enumerate(metrics):
                fig_compare.add_trace(go.Bar(
                    name=metric.replace('_', ' '),
                    x=estados_comparison['State'],
                    y=estados_comparison[metric],
                    marker_color=colors_compare[i],
                    text=estados_comparison[metric].round(2),
                    textposition='outside',
                    texttemplate='%{text:.2f}'
                ))
            
            fig_compare.update_layout(
                title="Comparación de Métricas entre Estados Seleccionados",
                xaxis_title="Estado",
                yaxis_title="Valor",
                barmode='group',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Tabla detallada
            st.markdown("#### 📋 Detalle Comparativo")
            st.dataframe(
                estados_comparison.style.format({
                    'Precio_Promedio': '${:.2f}',
                    'Precio_Min': '${:.2f}',
                    'Precio_Max': '${:.2f}',
                    'Std': '${:.2f}',
                    'CV': '{:.2f}%',
                    'Rango': '${:.2f}',
                    'Registros': '{:,}'
                }).background_gradient(subset=['Precio_Promedio'], cmap='RdYlGn_r'),
                use_container_width=True
            )

# TAB 5
with tab5:
    st.markdown("## 💰 Análisis de Poder Adquisitivo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_compra = go.Figure()
        fig_compra.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Kg_Tortilla_x_Salario'],
            fill='tozeroy',
            line=dict(color=COLORS['success'], width=2),
            name='kg/salario',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.0f} kg<extra></extra>'
        ))
        fig_compra.update_layout(
            title='Capacidad de Compra (kg con Salario Mensual)',
            xaxis_title='Fecha',
            yaxis_title='Kilogramos',
            height=400
        )
        st.plotly_chart(fig_compra, use_container_width=True)
    
    with col2:
        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Pct_Salario_en_Tortilla'],
            fill='tozeroy',
            line=dict(color=COLORS['accent'], width=2),
            name='% salario',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra></extra>'
        ))
        fig_pct.update_layout(
            title=f'% Salario en Tortilla (Hogar {PERSONAS_HOGAR} personas)',
            xaxis_title='Fecha',
            yaxis_title='Porcentaje',
            height=400
        )
        st.plotly_chart(fig_pct, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Indice_Accesibilidad'],
            line=dict(color=COLORS['primary'], width=2),
            name='Índice',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.1f}/100<extra></extra>'
        ))
        fig_acc.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Base 2007")
        fig_acc.update_layout(
            title='Índice de Accesibilidad (2007=100)',
            xaxis_title='Fecha',
            yaxis_title='Índice',
            height=400
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col4:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Indice_Salario'],
            name='Salario',
            line=dict(color=COLORS['success'], width=2)
        ))
        fig_comp.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Indice_Tortilla'],
            name='Tortilla',
            line=dict(color=COLORS['primary'], width=2)
        ))
        fig_comp.add_hline(y=100, line_dash="dash", line_color="gray")
        fig_comp.update_layout(
            title='Crecimiento Salario vs Tortilla',
            xaxis_title='Fecha',
            yaxis_title='Índice (2007=100)',
            height=400
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("### 📊 Resumen de Poder Adquisitivo")
    col1, col2, col3, col4 = st.columns(4)
    
    inicio_pa = df_filtered.iloc[0]
    fin_pa = df_filtered.iloc[-1]
    
    with col1:
        cambio_capacidad = ((fin_pa['Kg_Tortilla_x_Salario'] / inicio_pa['Kg_Tortilla_x_Salario']) - 1) * 100
        st.metric("📈 Cambio Capacidad", f"{cambio_capacidad:+.1f}%")
    
    with col2:
        st.metric("🎯 Accesibilidad Actual", f"{fin_pa['Indice_Accesibilidad']:.1f}/100")
    
    with col3:
        cambio_salario = ((fin_pa['Indice_Salario'] / inicio_pa['Indice_Salario']) - 1) * 100
        st.metric("💵 Crecimiento Salario", f"{cambio_salario:+.1f}%")
    
    with col4:
        cambio_tortilla = ((fin_pa['Indice_Tortilla'] / inicio_pa['Indice_Tortilla']) - 1) * 100
        st.metric("🌮 Crecimiento Tortilla", f"{cambio_tortilla:+.1f}%")

# TAB 6
with tab6:
    st.markdown("## 🌽 Análisis de Correlaciones Económicas")
    
    # ====================================================================
    # SECCIÓN 1: TORTILLA vs MAÍZ
    # ====================================================================
    st.markdown("### 1️⃣ Correlación Tortilla - Maíz (INPP)")
    
    st.markdown("""
    El maíz es la materia prima principal de la tortilla. Analizamos qué tan fuerte 
    es la relación entre el Índice Nacional de Precios al Productor (INPP) del maíz 
    y el precio final de la tortilla.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Análisis de Regresión")
        
        df_clean = df_master[['INPP_Maiz', 'Precio_Tortilla']].dropna()
        X = df_clean[['INPP_Maiz']].values
        y = df_clean['Precio_Tortilla'].values
        
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        corr_tortilla_maiz = df_master[['Precio_Tortilla', 'INPP_Maiz']].corr().iloc[0, 1]
        
        # Scatter plot con regresión
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=df_clean['INPP_Maiz'],
            y=df_clean['Precio_Tortilla'],
            mode='markers',
            name='Datos',
            marker=dict(color=COLORS['primary'], size=6, opacity=0.6),
            hovertemplate='INPP Maíz: %{x:.2f}<br>Precio: $%{y:.2f}/kg<extra></extra>'
        ))
        
        fig_scatter.add_trace(go.Scatter(
            x=x_range.flatten(),
            y=y_pred,
            mode='lines',
            name='Regresión Lineal',
            line=dict(color=COLORS['secondary'], width=3, dash='dash')
        ))
        
        fig_scatter.update_layout(
            title=f'Correlación Tortilla-Maíz (r={corr_tortilla_maiz:.3f}, R²={r2:.3f})',
            xaxis_title='INPP Maíz',
            yaxis_title='Precio Tortilla (MXN/kg)',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Métricas
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Correlación", f"{corr_tortilla_maiz:.4f}")
        with col_m2:
            st.metric("R²", f"{r2:.4f}")
        with col_m3:
            st.metric("Coeficiente", f"{model.coef_[0]:.4f}")
        
        st.info(f"""
        **Interpretación:**
        - Por cada punto que sube el INPP del maíz, la tortilla aumenta **${model.coef_[0]:.4f}/kg**
        - El INPP explica el **{r2*100:.1f}%** de la variación en precios de tortilla
        - Correlación **{'ALTA' if abs(corr_tortilla_maiz) > 0.7 else 'MODERADA' if abs(corr_tortilla_maiz) > 0.5 else 'BAJA'}**
        """)
    
    with col2:
        st.markdown("#### 📈 Series Temporales Comparativas")
        
        fig_series = go.Figure()
        
        fig_series.add_trace(go.Scatter(
            x=df_master['Date'],
            y=df_master['Indice_Tortilla'],
            name='Tortilla',
            line=dict(color=COLORS['primary'], width=2.5),
            hovertemplate='<b>Tortilla</b><br>%{x|%Y-%m}<br>Índice: %{y:.1f}<extra></extra>'
        ))
        
        fig_series.add_trace(go.Scatter(
            x=df_master['Date'],
            y=df_master['Indice_Maiz'],
            name='Maíz (INPP)',
            line=dict(color=COLORS['accent'], width=2.5),
            hovertemplate='<b>Maíz</b><br>%{x|%Y-%m}<br>Índice: %{y:.1f}<extra></extra>'
        ))
        
        fig_series.add_hline(y=100, line_dash="dash", line_color="gray", 
                            annotation_text="Base 2007=100")
        
        fig_series.update_layout(
            title='Índices Comparativos (Base 2007=100)',
            xaxis_title='Fecha',
            yaxis_title='Índice',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_series, use_container_width=True)
        
        # Crecimiento comparativo
        inicio_maiz = df_master['Indice_Maiz'].iloc[0]
        fin_maiz = df_master['Indice_Maiz'].iloc[-1]
        crecimiento_maiz = ((fin_maiz / inicio_maiz) - 1) * 100
        
        inicio_tortilla = df_master['Indice_Tortilla'].iloc[0]
        fin_tortilla = df_master['Indice_Tortilla'].iloc[-1]
        crecimiento_tortilla = ((fin_tortilla / inicio_tortilla) - 1) * 100
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Crecimiento Maíz", f"{crecimiento_maiz:+.1f}%")
        with col_c2:
            st.metric("Crecimiento Tortilla", f"{crecimiento_tortilla:+.1f}%")
    
    st.markdown("---")
    
    # ====================================================================
    # SECCIÓN 2: TORTILLA vs CANASTA ALIMENTARIA
    # ====================================================================
    st.markdown("### 2️⃣ Comparación Tortilla vs Canasta Alimentaria Urbana")
    
    st.markdown("""
    Comparamos la evolución del precio de la tortilla con la **Canasta Alimentaria Urbana** (CONEVAL),
    que representa el conjunto de alimentos básicos necesarios para satisfacer las necesidades
    nutricionales mínimas de la población.
    """)
    
    if 'Canasta_Urbana' in df_master.columns:
        df_canasta_clean = df_master[['Date', 'Precio_Tortilla', 'Canasta_Urbana', 
                                       'Indice_Tortilla', 'Indice_Canasta']].dropna()
        
        if len(df_canasta_clean) > 0:
            inicio_canasta = df_canasta_clean.iloc[0]
            fin_canasta = df_canasta_clean.iloc[-1]
            
            crecimiento_tortilla_pct = ((fin_canasta['Precio_Tortilla'] / inicio_canasta['Precio_Tortilla']) - 1) * 100
            crecimiento_canasta_pct = ((fin_canasta['Canasta_Urbana'] / inicio_canasta['Canasta_Urbana']) - 1) * 100
            diferencia_pct = crecimiento_tortilla_pct - crecimiento_canasta_pct
            
            st.markdown("#### 📊 Resumen Comparativo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📅 Periodo",
                    f"{inicio_canasta['Date'].year}-{fin_canasta['Date'].year}",
                    f"{len(df_canasta_clean)} meses"
                )
            
            with col2:
                st.metric(
                    "🌮 Crecimiento Tortilla",
                    f"{crecimiento_tortilla_pct:+.1f}%",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "🛒 Crecimiento Canasta",
                    f"{crecimiento_canasta_pct:+.1f}%",
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "📏 Diferencia",
                    f"{diferencia_pct:+.1f} pp",
                    help="Puntos porcentuales de diferencia",
                    delta_color="inverse"
                )
            
            # Interpretación
            if diferencia_pct > 5:
                st.warning(f"""
                ⚠️ **La tortilla creció {abs(diferencia_pct):.1f} puntos porcentuales MÁS que la canasta alimentaria**
                
                Esto sugiere que la tortilla ha experimentado un encarecimiento relativo mayor al conjunto 
                de alimentos básicos, afectando desproporcionadamente el presupuesto alimentario 
                de los hogares más vulnerables.
                """)
            elif diferencia_pct < -5:
                st.success(f"""
                ✅ **La tortilla creció {abs(diferencia_pct):.1f} puntos porcentuales MENOS que la canasta alimentaria**
                
                La tortilla ha mantenido un crecimiento más moderado que el conjunto de alimentos 
                básicos, representando un alivio relativo para el gasto alimentario.
                """)
            else:
                st.info(f"""
                ℹ️ **La tortilla y la canasta crecieron de manera similar** (diferencia: {abs(diferencia_pct):.1f} pp)
                
                Los precios de la tortilla han evolucionado en línea con la canasta alimentaria general.
                """)
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.markdown("#### 📈 Evolución de Índices")
                
                fig_indices_canasta = go.Figure()
                
                fig_indices_canasta.add_trace(go.Scatter(
                    x=df_canasta_clean['Date'],
                    y=df_canasta_clean['Indice_Tortilla'],
                    name='Tortilla',
                    line=dict(color=COLORS['primary'], width=3),
                    hovertemplate='<b>Tortilla</b><br>%{x|%Y-%m}<br>Índice: %{y:.1f}<extra></extra>'
                ))
                
                fig_indices_canasta.add_trace(go.Scatter(
                    x=df_canasta_clean['Date'],
                    y=df_canasta_clean['Indice_Canasta'],
                    name='Canasta Urbana',
                    line=dict(color=COLORS['success'], width=3),
                    hovertemplate='<b>Canasta</b><br>%{x|%Y-%m}<br>Índice: %{y:.1f}<extra></extra>'
                ))
                
                fig_indices_canasta.add_hline(
                    y=100, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="Base 2007=100"
                )
                
                if mostrar_bandas:
                    shapes = []
                    for nombre, info in PRESIDENTES.items():
                        inicio_pres = pd.Timestamp(info['inicio'])
                        fin_pres = pd.Timestamp(info['fin'])
                        
                        shapes.append(dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=inicio_pres,
                            x1=fin_pres,
                            y0=0,
                            y1=1,
                            fillcolor=info['color'],
                            opacity=0.08,
                            layer="below",
                            line_width=0
                        ))
                    
                    fig_indices_canasta.update_layout(shapes=shapes)
                
                fig_indices_canasta.update_layout(
                    title='Índices Base 2007=100',
                    xaxis_title='Fecha',
                    yaxis_title='Índice',
                    height=450,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_indices_canasta, use_container_width=True)
            
            with col_g2:
                st.markdown("#### 📊 Brecha Acumulada")
                
                df_canasta_clean['Brecha'] = df_canasta_clean['Indice_Tortilla'] - df_canasta_clean['Indice_Canasta']
                
                fig_brecha = go.Figure()
                
                fig_brecha.add_trace(go.Scatter(
                    x=df_canasta_clean['Date'],
                    y=df_canasta_clean['Brecha'],
                    fill='tozeroy',
                    line=dict(color=COLORS['primary'], width=2.5),
                    fillcolor='rgba(255, 107, 107, 0.3)',
                    name='Brecha',
                    hovertemplate='<b>Brecha</b><br>%{x|%Y-%m}<br>%{y:+.1f} puntos<extra></extra>'
                ))
                
                fig_brecha.add_hline(
                    y=0, 
                    line_dash="solid", 
                    line_color="black",
                    line_width=2
                )
                
                fig_brecha.update_layout(
                    title='Brecha Tortilla - Canasta<br>(Valores positivos: tortilla creció más)',
                    xaxis_title='Fecha',
                    yaxis_title='Diferencia en Índice',
                    height=450,
                    hovermode='x'
                )
                
                st.plotly_chart(fig_brecha, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🔍 Análisis por Administración Presidencial")
            
            resultados_pres_canasta = []
            
            for nombre, info in PRESIDENTES.items():
                inicio_pres = pd.Timestamp(info['inicio'])
                fin_pres = pd.Timestamp(info['fin'])
                
                mask = (df_canasta_clean['Date'] >= inicio_pres) & (df_canasta_clean['Date'] <= fin_pres)
                datos_periodo = df_canasta_clean[mask]
                
                if len(datos_periodo) > 0:
                    precio_tortilla_inicial = datos_periodo['Precio_Tortilla'].iloc[0]
                    precio_tortilla_final = datos_periodo['Precio_Tortilla'].iloc[-1]
                    crec_tortilla = ((precio_tortilla_final / precio_tortilla_inicial) - 1) * 100
                    
                    canasta_inicial = datos_periodo['Canasta_Urbana'].iloc[0]
                    canasta_final = datos_periodo['Canasta_Urbana'].iloc[-1]
                    crec_canasta = ((canasta_final / canasta_inicial) - 1) * 100
                    
                    resultados_pres_canasta.append({
                        'Presidente': nombre,
                        'Partido': info['partido'],
                        'Tortilla': crec_tortilla,
                        'Canasta': crec_canasta,
                        'Diferencia': crec_tortilla - crec_canasta,
                        'Color': info['color']
                    })
            
            df_pres_canasta = pd.DataFrame(resultados_pres_canasta)
            
            fig_pres_canasta = go.Figure()
            
            for i, row in df_pres_canasta.iterrows():
                fig_pres_canasta.add_trace(go.Bar(
                    name=row['Presidente'],
                    x=['Tortilla', 'Canasta'],
                    y=[row['Tortilla'], row['Canasta']],
                    marker_color=row['Color'],
                    text=[f"{row['Tortilla']:.1f}%", f"{row['Canasta']:.1f}%"],
                    textposition='outside',
                    hovertemplate=f"<b>{row['Presidente']}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>"
                ))
            
            fig_pres_canasta.update_layout(
                title='Crecimiento Tortilla vs Canasta por Presidente',
                yaxis_title='Crecimiento (%)',
                barmode='group',
                height=500,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig_pres_canasta, use_container_width=True)
            
            st.markdown("##### 📋 Tabla Resumen por Administración")
            
            st.dataframe(
                df_pres_canasta[['Presidente', 'Partido', 'Tortilla', 'Canasta', 'Diferencia']].style.format({
                    'Tortilla': '{:+.2f}%',
                    'Canasta': '{:+.2f}%',
                    'Diferencia': '{:+.2f} pp'
                }).background_gradient(subset=['Diferencia'], cmap='RdYlGn_r'),
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("💾 Exportar Análisis Tortilla-Canasta", key="export_canasta"):
                export_data = df_canasta_clean[['Date', 'Precio_Tortilla', 'Canasta_Urbana', 
                                                'Indice_Tortilla', 'Indice_Canasta', 'Brecha']]
                path = export_results_to_csv(export_data, "analisis_tortilla_canasta.csv")
                st.success(f"✅ Exportado: {path}")
        else:
            st.warning("⚠️ No hay suficientes datos para el análisis Tortilla-Canasta")
    else:
        st.warning("""
        ⚠️ **Columna 'Canasta_Urbana' no disponible**
        
        Para este análisis se requiere el archivo de Líneas de Pobreza de CONEVAL.
        Verifica que esté correctamente cargado.
        """)
    
    st.markdown("---")
    
    # ====================================================================
    # SECCIÓN 3: TORTILLA vs ITLP
    # ====================================================================
    if len(df_itlp_nacional) > 0:
        st.markdown("### 3️⃣ Correlación Tortilla - ITLP (Informalidad Laboral)")
        
        st.markdown("""
        La **Tasa de Informalidad Laboral (ITLP)** mide el porcentaje de población ocupada 
        que labora sin seguridad social. Exploramos si existe relación entre la informalidad 
        y los precios de la tortilla.
        """)
        
        df_master['YearQuarter'] = df_master['Date'].dt.to_period('Q')
        tortilla_trimestral = df_master.groupby('YearQuarter')['Precio_Tortilla'].mean().reset_index()
        tortilla_trimestral['Date'] = tortilla_trimestral['YearQuarter'].dt.to_timestamp()
        
        df_itlp_nacional['YearQuarter'] = df_itlp_nacional['Date'].dt.to_period('Q')
        df_tortilla_itlp = tortilla_trimestral.merge(
            df_itlp_nacional[['YearQuarter', 'ITLP']],
            on='YearQuarter',
            how='inner'
        )
        
        if len(df_tortilla_itlp) > 0:
            corr_itlp = df_tortilla_itlp[['Precio_Tortilla', 'ITLP']].corr().iloc[0, 1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Scatter Plot")
                
                df_itlp_clean = df_tortilla_itlp[['ITLP', 'Precio_Tortilla']].dropna()
                X_itlp = df_itlp_clean[['ITLP']].values
                y_itlp = df_itlp_clean['Precio_Tortilla'].values
                
                model_itlp = LinearRegression()
                model_itlp.fit(X_itlp, y_itlp)
                r2_itlp = model_itlp.score(X_itlp, y_itlp)
                
                x_itlp_range = np.linspace(X_itlp.min(), X_itlp.max(), 100)
                y_itlp_pred = model_itlp.predict(x_itlp_range.reshape(-1, 1))
                
                fig_itlp_scatter = go.Figure()
                
                fig_itlp_scatter.add_trace(go.Scatter(
                    x=df_itlp_clean['ITLP'],
                    y=df_itlp_clean['Precio_Tortilla'],
                    mode='markers',
                    name='Datos',
                    marker=dict(color=COLORS['primary'], size=8, opacity=0.6),
                    hovertemplate='ITLP: %{x:.2f}%<br>Precio: $%{y:.2f}/kg<extra></extra>'
                ))
                
                fig_itlp_scatter.add_trace(go.Scatter(
                    x=x_itlp_range.flatten(),
                    y=y_itlp_pred,
                    mode='lines',
                    name='Tendencia',
                    line=dict(color=COLORS['secondary'], width=3, dash='dash')
                ))
                
                fig_itlp_scatter.update_layout(
                    title=f'Correlación (r={corr_itlp:.3f}, R²={r2_itlp:.3f})',
                    xaxis_title='ITLP (%)',
                    yaxis_title='Precio Tortilla (MXN/kg)',
                    height=400
                )
                
                st.plotly_chart(fig_itlp_scatter, use_container_width=True)
                
                st.info(f"""
                **Interpretación:**
                - Correlación: **{corr_itlp:.3f}** ({'positiva' if corr_itlp > 0 else 'negativa'})
                - R²: **{r2_itlp:.3f}**
                - Relación: **{'FUERTE' if abs(corr_itlp) > 0.7 else 'MODERADA' if abs(corr_itlp) > 0.5 else 'DÉBIL'}**
                """)
            
            with col2:
                st.markdown("#### 📈 Series Temporales")
                
                fig_itlp_series = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_itlp_series.add_trace(
                    go.Scatter(
                        x=df_tortilla_itlp['Date'], 
                        y=df_tortilla_itlp['Precio_Tortilla'],
                        name='Tortilla', 
                        line=dict(color=COLORS['primary'], width=2.5)
                    ),
                    secondary_y=False
                )
                
                fig_itlp_series.add_trace(
                    go.Scatter(
                        x=df_tortilla_itlp['Date'], 
                        y=df_tortilla_itlp['ITLP'],
                        name='ITLP', 
                        line=dict(color=COLORS['secondary'], width=2.5)
                    ),
                    secondary_y=True
                )
                
                fig_itlp_series.update_xaxes(title_text="Fecha")
                fig_itlp_series.update_yaxes(title_text="Precio Tortilla (MXN/kg)", secondary_y=False)
                fig_itlp_series.update_yaxes(title_text="ITLP (%)", secondary_y=True)
                fig_itlp_series.update_layout(
                    title="Evolución Tortilla vs ITLP", 
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_itlp_series, use_container_width=True)
            
            # Conclusión
            if abs(corr_itlp) > 0.5:
                direccion = "positiva" if corr_itlp > 0 else "negativa"
                st.success(f"""
                ✅ **Correlación {direccion} {'fuerte' if abs(corr_itlp) > 0.7 else 'moderada'}**
                
                {'A mayor informalidad laboral, se observan mayores precios de tortilla.' if corr_itlp > 0 else 'A mayor informalidad laboral, se observan menores precios de tortilla.'}
                """)
            else:
                st.info("""
                ℹ️ **Correlación débil**
                
                No se observa una relación clara entre la informalidad laboral y 
                los precios de la tortilla. Otros factores pueden ser más determinantes.
                """)
        else:
            st.warning("⚠️ No hay suficientes datos para el análisis ITLP")
    else:
        st.info("ℹ️ Datos de ITLP no disponibles")
    
    st.markdown("---")
    
    # ====================================================================
    # RESUMEN DE CORRELACIONES
    # ====================================================================
    st.markdown("### 📊 Resumen de Correlaciones")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fuerza_maiz = "ALTA" if abs(corr_tortilla_maiz) > 0.7 else "MODERADA" if abs(corr_tortilla_maiz) > 0.5 else "BAJA"
        st.metric(
            "🌽 Tortilla - Maíz",
            f"{corr_tortilla_maiz:.3f}",
            f"Fuerza: {fuerza_maiz}"
        )
    
    with col2:
        if 'Canasta_Urbana' in df_master.columns and 'df_canasta_clean' in locals() and len(df_canasta_clean) > 0:
            st.metric(
                "🛒 Diferencia Canasta",
                f"{diferencia_pct:+.1f} pp",
                f"{'Tortilla mayor' if diferencia_pct > 0 else 'Tortilla menor'}"
            )
        else:
            st.metric("🛒 Canasta", "N/A", "Sin datos")
    
    with col3:
        if len(df_itlp_nacional) > 0 and 'df_tortilla_itlp' in locals() and len(df_tortilla_itlp) > 0:
            fuerza_itlp = "ALTA" if abs(corr_itlp) > 0.7 else "MODERADA" if abs(corr_itlp) > 0.5 else "BAJA"
            st.metric(
                "💼 Tortilla - ITLP",
                f"{corr_itlp:.3f}",
                f"Fuerza: {fuerza_itlp}"
            )
        else:
            st.metric("💼 ITLP", "N/A", "Sin datos")
    
    # Matriz de correlación visual
    st.markdown("---")
    st.markdown("### 🎯 Matriz de Correlaciones")
    
    # Preparar datos para matriz
    variables_corr = ['Precio_Tortilla', 'INPP_Maiz', 'Salario_Minimo']
    if 'Canasta_Urbana' in df_master.columns:
        variables_corr.append('Canasta_Urbana')
    
    df_corr_matrix = df_master[variables_corr].dropna()
    
    if len(df_corr_matrix) > 0:
        corr_matrix = df_corr_matrix.corr()
        
        # Labels más legibles
        labels_map = {
            'Precio_Tortilla': 'Tortilla',
            'INPP_Maiz': 'Maíz',
            'Salario_Minimo': 'Salario',
            'Canasta_Urbana': 'Canasta'
        }
        
        corr_matrix.index = [labels_map.get(x, x) for x in corr_matrix.index]
        corr_matrix.columns = [labels_map.get(x, x) for x in corr_matrix.columns]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 14},
            colorbar=dict(title="Correlación")
        ))
        
        fig_heatmap.update_layout(
            title='Matriz de Correlaciones entre Variables Económicas',
            height=500,
            xaxis_title="",
            yaxis_title=""
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.info("""
        **Interpretación de la matriz:**
        - Valores cercanos a **1**: Correlación positiva fuerte
        - Valores cercanos a **0**: Sin correlación
        - Valores cercanos a **-1**: Correlación negativa fuerte
        """)
    
    # Exportar análisis de correlaciones
    st.markdown("---")
    if st.button("💾 Exportar Análisis de Correlaciones Completo", key="export_correlaciones"):
        correlaciones_export = {
            'tortilla_maiz': {
                'correlacion': float(corr_tortilla_maiz),
                'r2': float(r2),
                'coeficiente': float(model.coef_[0])
            }
        }
        
        if 'diferencia_pct' in locals():
            correlaciones_export['tortilla_canasta'] = {
                'crecimiento_tortilla': float(crecimiento_tortilla_pct),
                'crecimiento_canasta': float(crecimiento_canasta_pct),
                'diferencia_pp': float(diferencia_pct)
            }
        
        if 'corr_itlp' in locals():
            correlaciones_export['tortilla_itlp'] = {
                'correlacion': float(corr_itlp),
                'r2': float(r2_itlp)
            }
        
        path = export_results_to_json(correlaciones_export, "analisis_correlaciones.json")
        st.success(f"✅ Exportado: {path}")
        

# TAB 7
with tab7:
    st.markdown("## 🔮 Proyecciones a 12 Meses")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Modelo 24m:** R²={proyecciones['r2_24m']:.3f} | Pendiente=${proyecciones['slope_24m']:+.2f}/año")
    with col2:
        st.info(f"**Modelo 12m:** R²={proyecciones['r2_12m']:.3f} | Pendiente=${proyecciones['slope_12m']:+.2f}/año")
    with col3:
        st.info(f"**Modelo Conservador:** Inflación 4% anual constante")
    
    fig_proj = go.Figure()
    
    historical = df_master.tail(36).copy()
    
    fig_proj.add_trace(go.Scatter(
        x=historical['Date'],
        y=historical['Precio_Tortilla'],
        mode='lines+markers',
        name='Histórico',
        line=dict(color='cyan', width=3),
        marker=dict(size=4)
    ))
    
    last_date = df_master['Date'].max()
    
    fig_proj.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        line=dict(
            color="red",
            width=3,
            dash="dot"
        )
    )
    
    fig_proj.add_annotation(
        x=last_date,
        y=1,
        xref="x",
        yref="paper",
        text="HOY",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="red", size=12, family="Arial Black")
    )
    
    future_dates = proyecciones['future_dates']
    y_pred_24m = proyecciones['y_pred_24m']
    y_pred_12m = proyecciones['y_pred_12m']
    y_pred_conservative = proyecciones['y_pred_conservative']
    std_12m = proyecciones['std_12m']
    
    fig_proj.add_trace(go.Scatter(
        x=future_dates,
        y=y_pred_24m,
        mode='lines+markers',
        name='Proyección 24m (Recomendada)',
        line=dict(color=COLORS['primary'], width=3, dash='dash'),
        marker=dict(size=6, symbol='square')
    ))
    
    fig_proj.add_trace(go.Scatter(
        x=future_dates,
        y=y_pred_12m,
        mode='lines+markers',
        name='Proyección 12m',
        line=dict(color=COLORS['secondary'], width=2.5, dash='dash'),
        marker=dict(size=5, symbol='triangle-up')
    ))
    
    fig_proj.add_trace(go.Scatter(
        x=future_dates,
        y=y_pred_conservative,
        mode='lines+markers',
        name='Proyección Conservadora',
        line=dict(color=COLORS['success'], width=2.5, dash='dot'),
        marker=dict(size=5, symbol='circle')
    ))
    
    fig_proj.add_trace(go.Scatter(
        x=future_dates,
        y=y_pred_24m + std_12m,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig_proj.add_trace(go.Scatter(
        x=future_dates,
        y=y_pred_24m - std_12m,
        mode='lines',
        fill='tonexty',
        line=dict(width=0),
        name=f'Banda Confianza (±${std_12m:.2f})',
        fillcolor='rgba(255, 107, 107, 0.2)',
        hoverinfo='skip'
    ))
    
    fig_proj.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=last_date,
        x1=future_dates[-1],
        y0=0,
        y1=1,
        fillcolor="yellow",
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    fig_proj.update_layout(
        title='Proyección a 12 Meses - Múltiples Escenarios',
        xaxis_title='Fecha',
        yaxis_title='Precio (MXN/kg)',
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_proj, use_container_width=True)
    
    st.markdown("### 📊 Resultados de Proyección")
    
    precio_actual = df_master['Precio_Tortilla'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💵 Precio Actual", f"${precio_actual:.2f}")
    
    with col2:
        proj_24m = y_pred_24m[-1]
        cambio_24m = ((proj_24m / precio_actual) - 1) * 100
        st.metric("🎯 Proyección 24m", f"${proj_24m:.2f}", f"{cambio_24m:+.1f}%")
    
    with col3:
        proj_12m = y_pred_12m[-1]
        cambio_12m = ((proj_12m / precio_actual) - 1) * 100
        st.metric("📊 Proyección 12m", f"${proj_12m:.2f}", f"{cambio_12m:+.1f}%")
    
    with col4:
        proj_cons = y_pred_conservative[-1]
        cambio_cons = ((proj_cons / precio_actual) - 1) * 100
        st.metric("🛡️ Conservadora", f"${proj_cons:.2f}", f"{cambio_cons:+.1f}%")
    
    st.markdown("### 📅 Proyecciones Mensuales Detalladas")
    
    df_proj_table = pd.DataFrame({
        'Mes': pd.to_datetime(future_dates).strftime('%Y-%m'),
        'Proyección 24m': [f"${x:.2f}" for x in y_pred_24m],
        'Proyección 12m': [f"${x:.2f}" for x in y_pred_12m],
        'Conservadora': [f"${x:.2f}" for x in y_pred_conservative],
        'Banda Superior': [f"${x:.2f}" for x in (y_pred_24m + std_12m)],
        'Banda Inferior': [f"${x:.2f}" for x in (y_pred_24m - std_12m)]
    })
    
    st.dataframe(df_proj_table, use_container_width=True)
    
    if st.button("💾 Exportar Proyecciones", key="export_proj"):
        df_proj_export = pd.DataFrame({
            'Fecha': future_dates,
            'Proyeccion_24m': y_pred_24m,
            'Proyeccion_12m': y_pred_12m,
            'Proyeccion_Conservadora': y_pred_conservative,
            'Banda_Superior': y_pred_24m + std_12m,
            'Banda_Inferior': y_pred_24m - std_12m
        })
        path = export_results_to_csv(df_proj_export, "proyecciones_12_meses.csv")
        st.success(f"✅ Exportado: {path}")

# ============================================================================
# REEMPLAZAR COMPLETAMENTE EL TAB 8 (ANÁLISIS POR CIUDADES)
# en dashboard_tortilla.py
# ============================================================================

with tab8:
    if HAS_CITIES:
        st.markdown("## 🏙️ Análisis Detallado por Ciudades")
        
        # ====================================================================
        # PREPARAR DATOS
        # ====================================================================
        price_col = "Price per kilogram"
        city_counts = df_tortilla["City"].value_counts()
        valid_cities = city_counts[city_counts >= 10].index
        df_cities = df_tortilla[df_tortilla["City"].isin(valid_cities)].copy()
        
        # Estadísticas generales por ciudad
        city_stats_general = df_cities.groupby("City").agg({
            price_col: ['mean', 'std', 'count'],
            'State': 'first'
        }).round(2)
        city_stats_general.columns = ['mean', 'std', 'count', 'Estado']
        city_stats_general = city_stats_general.sort_values('mean', ascending=False)
        
        st.info(f"📍 **{len(valid_cities)} ciudades** analizadas (con ≥10 registros)")
        
        # ====================================================================
        # SECCIÓN 1: TIMELINE INTERACTIVO POR AÑO
        # ====================================================================
        st.markdown("### 📅 Evolución Temporal de Ciudades")
        
        st.markdown("""
        Explora cómo han cambiado los rankings de precios a lo largo del tiempo.
        Selecciona un año para ver las ciudades más caras y baratas en ese periodo.
        """)
        
        # Preparar datos por año
        years_cities = sorted(df_cities['Year'].unique())
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_year_cities = st.select_slider(
                "Selecciona año para analizar:",
                options=years_cities,
                value=years_cities[-1],
                key="year_slider_cities"
            )
        
        with col2:
            top_n_cities = st.selectbox(
                "Ciudades a mostrar:",
                options=[10, 15, 20, 25, 30],
                index=2,  # 20 por defecto
                key="top_n_cities"
            )
        
        with col3:
            st.metric("📅 Año", selected_year_cities)
        
        # Filtrar datos del año seleccionado
        df_year_cities = df_cities[df_cities['Year'] == selected_year_cities].copy()
        
        if len(df_year_cities) > 0:
            # Calcular estadísticas del año
            city_stats_year = df_year_cities.groupby('City').agg({
                price_col: 'mean',
                'State': 'first'
            }).round(2)
            city_stats_year.columns = ['Precio', 'Estado']
            city_stats_year = city_stats_year.sort_values('Precio', ascending=False)
            
            # Obtener presidente del año
            presidente_year, partido_year = get_presidente_actual(f"{selected_year_cities}-06-01")
            
            if presidente_year:
                st.info(f"👤 **Presidente en {selected_year_cities}:** {presidente_year} ({partido_year})")
            
            # Visualización Top/Bottom del año
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### 🔴 Top {top_n_cities} MÁS CARAS en {selected_year_cities}")
                
                top_caras_year = city_stats_year.head(top_n_cities).reset_index()
                
                fig_caras_year = go.Figure()
                
                # Colores degradados en rojo
                colors_red = [f'rgb({int(255*(1-i/top_n_cities))}, {int(50+100*i/top_n_cities)}, {int(50+100*i/top_n_cities)})' 
                             for i in range(len(top_caras_year))]
                
                fig_caras_year.add_trace(go.Bar(
                    y=top_caras_year['City'][::-1],  # Invertir para mayor arriba
                    x=top_caras_year['Precio'][::-1],
                    orientation='h',
                    marker=dict(
                        color=colors_red[::-1],
                        line=dict(color='black', width=1)
                    ),
                    text=[f"${v:.2f}" for v in top_caras_year['Precio'][::-1]],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Estado: %{customdata}<br>Precio: $%{x:.2f}/kg<extra></extra>',
                    customdata=top_caras_year['Estado'][::-1]
                ))
                
                fig_caras_year.update_layout(
                    title=f"Ciudades Más Caras - {selected_year_cities}",
                    xaxis_title="Precio (MXN/kg)",
                    height=max(400, top_n_cities * 20),
                    showlegend=False,
                    plot_bgcolor='rgba(250,250,250,0.9)'
                )
                
                st.plotly_chart(fig_caras_year, use_container_width=True)
            
            with col2:
                st.markdown(f"#### 🟢 Top {top_n_cities} MÁS BARATAS en {selected_year_cities}")
                
                top_baratas_year = city_stats_year.tail(top_n_cities).sort_values('Precio').reset_index()
                
                fig_baratas_year = go.Figure()
                
                # Colores degradados en verde
                colors_green = [f'rgb({int(100+100*i/top_n_cities)}, {int(200-50*i/top_n_cities)}, {int(100+50*i/top_n_cities)})' 
                               for i in range(len(top_baratas_year))]
                
                fig_baratas_year.add_trace(go.Bar(
                    y=top_baratas_year['City'][::-1],
                    x=top_baratas_year['Precio'][::-1],
                    orientation='h',
                    marker=dict(
                        color=colors_green[::-1],
                        line=dict(color='black', width=1)
                    ),
                    text=[f"${v:.2f}" for v in top_baratas_year['Precio'][::-1]],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Estado: %{customdata}<br>Precio: $%{x:.2f}/kg<extra></extra>',
                    customdata=top_baratas_year['Estado'][::-1]
                ))
                
                fig_baratas_year.update_layout(
                    title=f"Ciudades Más Baratas - {selected_year_cities}",
                    xaxis_title="Precio (MXN/kg)",
                    height=max(400, top_n_cities * 20),
                    showlegend=False,
                    plot_bgcolor='rgba(250,250,250,0.9)'
                )
                
                st.plotly_chart(fig_baratas_year, use_container_width=True)
            
            # Métricas del año
            st.markdown(f"#### 📊 Estadísticas Globales {selected_year_cities}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            year_city_stats = city_stats_year['Precio'].describe()
            
            with col1:
                st.metric("🏙️ Ciudades", len(city_stats_year))
            with col2:
                st.metric("💵 Promedio", f"${year_city_stats['mean']:.2f}")
            with col3:
                st.metric("📉 Mínimo", f"${year_city_stats['min']:.2f}")
            with col4:
                st.metric("📈 Máximo", f"${year_city_stats['max']:.2f}")
            with col5:
                st.metric("📏 Brecha", f"${year_city_stats['max'] - year_city_stats['min']:.2f}")
            
            # Comparación interanual
            if selected_year_cities > years_cities[0]:
                st.markdown("---")
                st.markdown(f"#### 📈 Cambios vs {selected_year_cities - 1}")
                
                df_year_anterior = df_cities[df_cities['Year'] == selected_year_cities - 1].copy()
                
                if len(df_year_anterior) > 0:
                    city_stats_anterior = df_year_anterior.groupby('City')[price_col].mean()
                    
                    # Merge para comparar
                    comparison_cities = city_stats_year.copy()
                    comparison_cities['Precio_Anterior'] = comparison_cities.index.map(city_stats_anterior)
                    comparison_cities = comparison_cities.dropna(subset=['Precio_Anterior'])
                    comparison_cities['Cambio_Abs'] = comparison_cities['Precio'] - comparison_cities['Precio_Anterior']
                    comparison_cities['Cambio_Pct'] = (comparison_cities['Cambio_Abs'] / comparison_cities['Precio_Anterior']) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 📈 Top 10 Mayores Incrementos")
                        top_incrementos_cities = comparison_cities.nlargest(10, 'Cambio_Pct')[
                            ['Estado', 'Cambio_Pct', 'Cambio_Abs']
                        ].reset_index()
                        
                        st.dataframe(
                            top_incrementos_cities.style.format({
                                'Cambio_Pct': '{:+.2f}%',
                                'Cambio_Abs': '${:+.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col2:
                        st.markdown("##### 📉 Top 10 Menores Cambios")
                        top_decrementos_cities = comparison_cities.nsmallest(10, 'Cambio_Pct')[
                            ['Estado', 'Cambio_Pct', 'Cambio_Abs']
                        ].reset_index()
                        
                        st.dataframe(
                            top_decrementos_cities.style.format({
                                'Cambio_Pct': '{:+.2f}%',
                                'Cambio_Abs': '${:+.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
        
        else:
            st.warning(f"⚠️ No hay datos disponibles para el año {selected_year_cities}")
        
        st.markdown("---")
        
        # ====================================================================
        # SECCIÓN 2: COMPARADOR MULTI-CIUDAD (MEJORADO)
        # ====================================================================
        st.markdown("### 🔍 Comparador de Ciudades (Histórico Completo)")
        
        st.markdown("""
        Compara la evolución temporal de múltiples ciudades a lo largo de todo el periodo disponible.
        Selecciona hasta 10 ciudades para comparar.
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # CLAVE: usar key único y help para clarificar
            ciudades_comparar = st.multiselect(
                "Selecciona ciudades para comparar:",
                options=sorted(city_stats_general.index.tolist()),
                default=city_stats_general.head(5).index.tolist(),
                max_selections=10,
                key="multiselect_ciudades_comparar",
                help="Puedes seleccionar hasta 10 ciudades para ver su evolución histórica"
            )
        
        with col2:
            st.metric("🏙️ Ciudades", len(ciudades_comparar))
        
        if ciudades_comparar:
            # ================================================================
            # GRÁFICO DE EVOLUCIÓN TEMPORAL
            # ================================================================
            st.markdown("#### 📈 Evolución Temporal de Precios")
            
            fig_evolution_cities = go.Figure()
            
            # Colores variados
            colors_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Bold
            
            for i, city in enumerate(ciudades_comparar):
                city_data_temporal = df_cities[df_cities['City'] == city].groupby('Date')[price_col].mean()
                estado_city = city_stats_general.loc[city, 'Estado']
                
                fig_evolution_cities.add_trace(go.Scatter(
                    x=city_data_temporal.index,
                    y=city_data_temporal.values,
                    mode='lines+markers',
                    name=f"{city} ({estado_city})",
                    line=dict(width=2.5, color=colors_palette[i % len(colors_palette)]),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{city}</b><br>%{{x|%Y-%m}}<br>${{y:.2f}}/kg<extra></extra>'
                ))
            
            # Bandas presidenciales de fondo (opcional)
            if mostrar_bandas:
                shapes = []
                for nombre, info in PRESIDENTES.items():
                    inicio = pd.Timestamp(info['inicio'])
                    fin = pd.Timestamp(info['fin'])
                    
                    shapes.append(dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=inicio,
                        x1=fin,
                        y0=0,
                        y1=1,
                        fillcolor=info['color'],
                        opacity=0.08,
                        layer="below",
                        line_width=0
                    ))
                
                fig_evolution_cities.update_layout(shapes=shapes)
            
            fig_evolution_cities.update_layout(
                title=f"Comparativa Temporal - {len(ciudades_comparar)} Ciudades",
                xaxis_title="Fecha",
                yaxis_title="Precio (MXN/kg)",
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                )
            )
            
            st.plotly_chart(fig_evolution_cities, use_container_width=True)
            
            # ================================================================
            # ESTADÍSTICAS COMPARATIVAS
            # ================================================================
            st.markdown("#### 📊 Estadísticas Comparativas (Histórico)")
            
            comp_stats_cities = []
            for city in ciudades_comparar:
                city_data_full = df_cities[df_cities['City'] == city][price_col]
                estado_city = city_stats_general.loc[city, 'Estado']
                
                comp_stats_cities.append({
                    'Ciudad': city,
                    'Estado': estado_city,
                    'Promedio': city_data_full.mean(),
                    'Mínimo': city_data_full.min(),
                    'Máximo': city_data_full.max(),
                    'Volatilidad (%)': (city_data_full.std() / city_data_full.mean() * 100),
                    'Registros': len(city_data_full)
                })
            
            df_comp_cities = pd.DataFrame(comp_stats_cities).sort_values('Promedio', ascending=False)
            
            st.dataframe(
                df_comp_cities.style.format({
                    'Promedio': '${:.2f}',
                    'Mínimo': '${:.2f}',
                    'Máximo': '${:.2f}',
                    'Volatilidad (%)': '{:.2f}%',
                    'Registros': '{:,}'
                }).background_gradient(subset=['Promedio'], cmap='RdYlGn_r'),
                use_container_width=True,
                hide_index=True
            )
            
            # ================================================================
            # GRÁFICO DE BARRAS COMPARATIVO
            # ================================================================
            st.markdown("#### 📊 Comparación de Métricas")
            
            metrics_to_compare = st.multiselect(
                "Selecciona métricas a comparar:",
                options=['Promedio', 'Mínimo', 'Máximo', 'Volatilidad (%)'],
                default=['Promedio', 'Máximo'],
                key="metrics_compare_cities"
            )
            
            if metrics_to_compare:
                fig_compare_metrics = go.Figure()
                
                colors_metrics = {
                    'Promedio': '#FF6B6B',
                    'Mínimo': '#4ECDC4',
                    'Máximo': '#FFE66D',
                    'Volatilidad (%)': '#9B59B6'
                }
                
                for metric in metrics_to_compare:
                    fig_compare_metrics.add_trace(go.Bar(
                        name=metric,
                        x=df_comp_cities['Ciudad'],
                        y=df_comp_cities[metric],
                        marker_color=colors_metrics.get(metric, '#666'),
                        text=df_comp_cities[metric].round(2),
                        textposition='outside',
                        texttemplate='%{text:.2f}'
                    ))
                
                fig_compare_metrics.update_layout(
                    title="Comparación de Métricas entre Ciudades Seleccionadas",
                    xaxis_title="Ciudad",
                    yaxis_title="Valor",
                    barmode='group',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_compare_metrics, use_container_width=True)
        
        else:
            st.info("👆 Selecciona al menos una ciudad para ver la comparación")
        
        st.markdown("---")
        
        # ====================================================================
        # SECCIÓN 3: BÚSQUEDA Y EXPLORACIÓN GENERAL
        # ====================================================================
        st.markdown("### 🔎 Búsqueda y Exploración General")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_term = st.text_input(
                "🔎 Buscar ciudad:",
                placeholder="Ej: Guadalajara, Monterrey, CDMX...",
                key="search_cities_general"
            )
        
        with col2:
            sort_by = st.selectbox(
                "Ordenar por:",
                options=['mean', 'std', 'count'],
                format_func=lambda x: {'mean': 'Precio Promedio', 'std': 'Volatilidad', 'count': 'Registros'}[x],
                key="sort_cities"
            )
        
        # Filtrar y ordenar
        if search_term:
            filtered_cities = city_stats_general[
                city_stats_general.index.str.contains(search_term, case=False, na=False)
            ]
        else:
            filtered_cities = city_stats_general
        
        filtered_cities = filtered_cities.sort_values(sort_by, ascending=False)
        
        st.markdown(f"**{len(filtered_cities)} ciudades encontradas**")
        
        st.dataframe(
            filtered_cities.style.format({
                'mean': '${:.2f}',
                'std': '${:.2f}',
                'count': '{:.0f}'
            }).background_gradient(subset=['mean'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=400
        )
        
        # ====================================================================
        # SECCIÓN 4: EXPORTACIONES
        # ====================================================================
        st.markdown("---")
        st.markdown("### 💾 Exportar Datos de Ciudades")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Exportar Estadísticas Generales", key="export_cities_general"):
                city_export = city_stats_general.reset_index()
                city_export.columns = ['Ciudad', 'Precio_Promedio', 'Desv_Std', 'Registros', 'Estado']
                path = export_results_to_csv(city_export, "estadisticas_ciudades_general.csv")
                st.success(f"✅ Exportado: {path}")
                
                csv = city_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar CSV",
                    data=csv,
                    file_name="estadisticas_ciudades_general.csv",
                    mime="text/csv",
                    key="download_cities_general"
                )
        
        with col2:
            if st.button(f"📥 Exportar Datos {selected_year_cities}", key="export_cities_year"):
                if len(df_year_cities) > 0:
                    city_year_export = city_stats_year.reset_index()
                    city_year_export.columns = ['Ciudad', 'Precio', 'Estado']
                    city_year_export['Año'] = selected_year_cities
                    path = export_results_to_csv(city_year_export, f"precios_ciudades_{selected_year_cities}.csv")
                    st.success(f"✅ Exportado: {path}")
                    
                    csv_year = city_year_export.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Descargar CSV",
                        data=csv_year,
                        file_name=f"precios_ciudades_{selected_year_cities}.csv",
                        mime="text/csv",
                        key="download_cities_year"
                    )
                else:
                    st.warning("⚠️ No hay datos para exportar del año seleccionado")
    
    else:
        st.warning("⚠️ Datos de ciudades no disponibles en el dataset")
        st.info("""
        **El dataset cargado no contiene información por ciudad.**
        
        Esta funcionalidad requiere una columna 'City' en el archivo de datos.
        Verifica que tu archivo `tortilla_prices.csv` incluya esta información.
        """)
with tab9:
    st.markdown("## 📊 Análisis de Disparidad Regional")
    
    st.markdown("""
    Este análisis muestra cómo ha evolucionado la **brecha de precios** entre estados a lo largo del tiempo,
    revelando si los precios se están homogeneizando o divergiendo.
    """)
    
    # Calcular disparidad por año
    yearly_disparity = []
    price_col = "Price per kilogram"
    
    for year in sorted(df_tortilla['Year'].unique()):
        year_data = df_tortilla[df_tortilla['Year'] == year]
        state_means = year_data.groupby('State')[price_col].mean()
        
        yearly_disparity.append({
            'Year': year,
            'Max': state_means.max(),
            'Min': state_means.min(),
            'Range': state_means.max() - state_means.min(),
            'Std': state_means.std(),
            'CV': (state_means.std() / state_means.mean() * 100),
            'Mean': state_means.mean()
        })
    
    df_disparity = pd.DataFrame(yearly_disparity).sort_values('Year')
    
    # 4 visualizaciones en grid 2x2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📏 Rango de Precios (Max - Min)")
        
        fig_range = go.Figure()
        
        fig_range.add_trace(go.Scatter(
            x=df_disparity['Year'],
            y=df_disparity['Range'],
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8, symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)',
            name='Rango',
            hovertemplate='<b>%{x}</b><br>Rango: $%{y:.2f}/kg<extra></extra>'
        ))
        
        # Línea de tendencia
        z = np.polyfit(df_disparity['Year'], df_disparity['Range'], 1)
        p = np.poly1d(z)
        
        fig_range.add_trace(go.Scatter(
            x=df_disparity['Year'],
            y=p(df_disparity['Year']),
            mode='lines',
            line=dict(color='yellow', width=2, dash='dash'),
            name='Tendencia',
            hoverinfo='skip'
        ))
        
        fig_range.update_layout(
            title="Evolución del Rango de Precios entre Estados",
            xaxis_title="Año",
            yaxis_title="Rango (MXN/kg)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_range, use_container_width=True)
        
        # Interpretación
        tendencia_range = "aumentando" if z[0] > 0 else "disminuyendo"
        st.info(f"📈 La brecha entre estados está **{tendencia_range}** (pendiente: ${z[0]:.3f}/año)")
    
    with col2:
        st.markdown("### 📊 Desviación Estándar")
        
        fig_std = go.Figure()
        
        fig_std.add_trace(go.Scatter(
            x=df_disparity['Year'],
            y=df_disparity['Std'],
            mode='lines+markers',
            line=dict(color=COLORS['accent'], width=3),
            marker=dict(size=8, symbol='square'),
            fill='tozeroy',
            fillcolor='rgba(255, 230, 109, 0.2)',
            name='Desv. Std',
            hovertemplate='<b>%{x}</b><br>Std: $%{y:.2f}<extra></extra>'
        ))
        
        z_std = np.polyfit(df_disparity['Year'], df_disparity['Std'], 1)
        p_std = np.poly1d(z_std)
        
        fig_std.add_trace(go.Scatter(
            x=df_disparity['Year'],
            y=p_std(df_disparity['Year']),
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Tendencia',
            hoverinfo='skip'
        ))
        
        fig_std.update_layout(
            title="Variabilidad de Precios entre Estados",
            xaxis_title="Año",
            yaxis_title="Desviación Estándar (MXN/kg)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_std, use_container_width=True)
        
        tendencia_std = "aumentando" if z_std[0] > 0 else "disminuyendo"
        st.info(f"📊 La variabilidad está **{tendencia_std}** (pendiente: ${z_std[0]:.3f}/año)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 📉 Coeficiente de Variación (%)")
        
        fig_cv = go.Figure()
        
        fig_cv.add_trace(go.Scatter(
            x=df_disparity['Year'],
            y=df_disparity['CV'],
            mode='lines+markers',
            line=dict(color=COLORS['secondary'], width=3),
            marker=dict(size=8, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(78, 205, 196, 0.2)',
            name='CV',
            hovertemplate='<b>%{x}</b><br>CV: %{y:.2f}%<extra></extra>'
        ))
        
        fig_cv.add_hline(
            y=df_disparity['CV'].mean(),
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Promedio: {df_disparity['CV'].mean():.2f}%",
            annotation_position="right"
        )
        
        fig_cv.update_layout(
            title="Volatilidad Relativa entre Estados",
            xaxis_title="Año",
            yaxis_title="Coeficiente de Variación (%)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cv, use_container_width=True)
    
    with col4:
        st.markdown("### 🔺 Máximo vs Mínimo")
        
        fig_maxmin = go.Figure()
        
        fig_maxmin.add_trace(go.Scatter(
            x=df_disparity['Year'],
            y=df_disparity['Max'],
            mode='lines+markers',
            line=dict(color='red', width=2.5),
            marker=dict(size=7),
            name='Máximo',
            hovertemplate='<b>%{x}</b><br>Max: $%{y:.2f}<extra></extra>'
        ))
        
        fig_maxmin.add_trace(go.Scatter(
            x=df_disparity['Year'],
            y=df_disparity['Min'],
            mode='lines+markers',
            line=dict(color='green', width=2.5),
            marker=dict(size=7),
            name='Mínimo',
            hovertemplate='<b>%{x}</b><br>Min: $%{y:.2f}<extra></extra>'
        ))
        
        # Área sombreada entre max y min
        fig_maxmin.add_trace(go.Scatter(
            x=df_disparity['Year'].tolist() + df_disparity['Year'].tolist()[::-1],
            y=df_disparity['Max'].tolist() + df_disparity['Min'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(128,128,128,0.2)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_maxmin.update_layout(
            title="Evolución de Precios Extremos",
            xaxis_title="Año",
            yaxis_title="Precio (MXN/kg)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_maxmin, use_container_width=True)
    
    # Tabla resumen
    st.markdown("### 📋 Resumen por Año")
    
    st.dataframe(
        df_disparity.style.format({
            'Max': '${:.2f}',
            'Min': '${:.2f}',
            'Range': '${:.2f}',
            'Std': '${:.2f}',
            'CV': '{:.2f}%',
            'Mean': '${:.2f}'
        }).background_gradient(subset=['Range', 'CV'], cmap='Reds'),
        use_container_width=True
    )
    
    # Análisis de convergencia/divergencia
    st.markdown("### 🎯 Análisis de Convergencia Regional")
    
    col1, col2, col3 = st.columns(3)
    
    inicio_range = df_disparity.iloc[0]['Range']
    fin_range = df_disparity.iloc[-1]['Range']
    cambio_range = ((fin_range / inicio_range) - 1) * 100
    
    inicio_cv = df_disparity.iloc[0]['CV']
    fin_cv = df_disparity.iloc[-1]['CV']
    cambio_cv = fin_cv - inicio_cv
    
    with col1:
        st.metric("Cambio en Brecha", f"{cambio_range:+.1f}%", 
                 delta_color="inverse")
    
    with col2:
        st.metric("Cambio en CV", f"{cambio_cv:+.2f} pp",
                 delta_color="inverse")
    
    with col3:
        if cambio_range < 0 and cambio_cv < 0:
            convergencia = "✅ CONVERGENCIA"
            color = "success"
        elif cambio_range > 0 and cambio_cv > 0:
            convergencia = "⚠️ DIVERGENCIA"
            color = "warning"
        else:
            convergencia = "ℹ️ MIXTO"
            color = "info"
        
        st.metric("Tendencia", convergencia)
    
    if cambio_range < 0:
        st.success("""
        **✅ Señales de Convergencia:** Los precios entre estados se están homogeneizando.
        Esto puede indicar mejor integración de mercados y reducción de barreras comerciales.
        """)
    else:
        st.warning("""
        **⚠️ Señales de Divergencia:** La brecha de precios entre estados está aumentando.
        Esto puede reflejar diferencias crecientes en costos locales, poder adquisitivo o políticas regionales.
        """)
    
    # Exportar
    if st.button("💾 Exportar Análisis de Disparidad", key="export_disparity"):
        path = export_results_to_csv(df_disparity, "disparidad_regional.csv")
        st.success(f"✅ Exportado: {path}")


# TAB 10
with tab10:
    st.markdown("## 📄 Conclusiones del Análisis")
    
    precio_inicial = df_master['Precio_Tortilla'].iloc[0]
    precio_final = df_master['Precio_Tortilla'].iloc[-1]
    incremento_total = ((precio_final / precio_inicial) - 1) * 100
    
    acc_inicial = df_master['Indice_Accesibilidad'].iloc[0]
    acc_final = df_master['Indice_Accesibilidad'].iloc[-1]
    
    corr_maiz = df_master[['Precio_Tortilla', 'INPP_Maiz']].corr().iloc[0, 1]
    

    st.markdown("### 🎯 1. Evolución Histórica General")
    st.markdown(f"""
    **Periodo analizado:** {df_master['Date'].min().year} - {df_master['Date'].max().year}
    
    - **Incremento total:** {incremento_total:.1f}% (de ${precio_inicial:.2f} a ${precio_final:.2f}/kg)
    - **CAGR promedio:** {(((precio_final/precio_inicial)**(1/((df_master['Date'].max()-df_master['Date'].min()).days/365.25))-1)*100):.2f}% anual
    - **Volatilidad global:** {df_master['Precio_Tortilla'].pct_change().std() * np.sqrt(12) * 100:.2f}%
    
    El precio de la tortilla ha experimentado un crecimiento sostenido a lo largo del periodo analizado, 
    con momentos de alta volatilidad asociados a eventos macroeconómicos y cambios en políticas públicas.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    

    st.markdown("### 🏛️ 2. Impacto de Administraciones Presidenciales")
    
    presidente_mayor_incremento = df_pres.loc[df_pres['Incremento_Porcentual'].idxmax()]
    presidente_menor_incremento = df_pres.loc[df_pres['Incremento_Porcentual'].idxmin()]
    presidente_mayor_volatilidad = df_pres.loc[df_pres['Volatilidad'].idxmax()]
    
    st.markdown(f"""
    **Principales hallazgos por administración:**
    
    - **Mayor incremento:** {presidente_mayor_incremento['Presidente']} ({presidente_mayor_incremento['Partido']}) 
      con {presidente_mayor_incremento['Incremento_Porcentual']:.1f}% (CAGR: {presidente_mayor_incremento['CAGR']:.2f}%)
    
    - **Menor incremento:** {presidente_menor_incremento['Presidente']} ({presidente_menor_incremento['Partido']}) 
      con {presidente_menor_incremento['Incremento_Porcentual']:.1f}% (CAGR: {presidente_menor_incremento['CAGR']:.2f}%)
    
    - **Mayor volatilidad:** {presidente_mayor_volatilidad['Presidente']} con {presidente_mayor_volatilidad['Volatilidad']:.2f}%
    
    Las diferencias en los incrementos de precio reflejan no solo políticas internas, 
    sino también contextos económicos globales y eventos extraordinarios durante cada administración.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    

    st.markdown("### ⚡ 3. Impacto de Eventos Críticos")
    
    evento_mayor_impacto = df_eventos.loc[df_eventos['Cambio_Porcentual'].abs().idxmax()]
    eventos_positivos = df_eventos[df_eventos['Cambio_Porcentual'] > 0]
    eventos_negativos = df_eventos[df_eventos['Cambio_Porcentual'] < 0]
    
    st.markdown(f"""
    **Análisis de eventos con mayor impacto:**
    
    - **Evento con mayor impacto:** {evento_mayor_impacto['Evento']} ({evento_mayor_impacto['Fecha']})
      → Cambio de {evento_mayor_impacto['Cambio_Porcentual']:+.1f}%
    
    - **Eventos con impacto alcista:** {len(eventos_positivos)} eventos
      (promedio: {eventos_positivos['Cambio_Porcentual'].mean():+.1f}%)
    
    - **Eventos con impacto bajista:** {len(eventos_negativos)} eventos
      (promedio: {eventos_negativos['Cambio_Porcentual'].mean():+.1f}%)
    
    Los eventos económicos (crisis, inflación) tienen el mayor impacto en el precio de la tortilla,
    seguidos por eventos sanitarios. Los cambios políticos muestran efectos más graduales.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
 
    st.markdown("### 🗺️ 4. Disparidad Regional")
    
    estado_mas_caro = state_stats.index[0]
    estado_mas_barato = state_stats.index[-1]
    brecha_regional = state_stats.iloc[0]['mean'] - state_stats.iloc[-1]['mean']
    
    st.markdown(f"""
    **Hallazgos geográficos:**
    
    - **Estado más caro:** {estado_mas_caro} (${state_stats.iloc[0]['mean']:.2f}/kg)
    - **Estado más barato:** {estado_mas_barato} (${state_stats.iloc[-1]['mean']:.2f}/kg)
    - **Brecha regional:** ${brecha_regional:.2f}/kg ({(brecha_regional/state_stats.iloc[-1]['mean']*100):.1f}%)
    
    - **Coeficiente de variación promedio:** {state_stats['cv'].mean():.2f}%
    
    Existe una significativa disparidad regional en los precios, reflejando diferencias en:
    costos de transporte, competencia local, poder adquisitivo regional y políticas estatales.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

 
    st.markdown("### 🏙️ 5. Análisis Urbano")
    
    if HAS_CITIES:
        ciudad_mas_cara = city_stats_general.index[0]
        ciudad_mas_barata = city_stats_general.index[-1]
        brecha_urbana = city_stats_general.iloc[0]['mean'] - city_stats_general.iloc[-1]['mean']
        
        st.markdown(f"""
        **Hallazgos a nivel ciudad:**
        
        - **Ciudad más cara:** {ciudad_mas_cara} (${city_stats_general.iloc[0]['mean']:.2f}/kg) - {city_stats_general.iloc[0]['Estado']}
        - **Ciudad más barata:** {ciudad_mas_barata} (${city_stats_general.iloc[-1]['mean']:.2f}/kg) - {city_stats_general.iloc[-1]['Estado']}
    - **Brecha urbana:** ${brecha_urbana:.2f}/kg ({(brecha_urbana/city_stats_general.iloc[-1]['mean']*100):.1f}%)
        
        La disparidad a nivel ciudad es **{('mayor' if brecha_urbana/city_stats_general.iloc[-1]['mean'] > brecha_regional/state_stats.iloc[-1]['mean'] else 'menor')}** 
        que a nivel estatal, lo que sugiere una alta heterogeneidad intra-estatal.
        """)
    else:
        st.info("Datos de ciudades no disponibles")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    st.markdown("### 📐 6. Convergencia Regional")
    
    st.markdown(f"""
    **Análisis de homogeneización de precios:**
    
    - **Tendencia del rango:** {tendencia_range} ({cambio_range:+.1f}%)
    - **Tendencia del CV:** {('aumentando' if cambio_cv > 0 else 'disminuyendo')} ({cambio_cv:+.2f} pp)
    - **Estado actual:** {convergencia}
    
    {'Los mercados regionales muestran señales de integración, posiblemente debido a mejoras en infraestructura logística y menor fricción comercial.' if cambio_range < 0 else 'Persisten o aumentan las diferencias regionales, lo que puede reflejar políticas locales divergentes o diferencias estructurales en los mercados.'}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💾 Exportar Análisis Completo")
    
    if st.button("📥 Generar Reporte Completo (JSON)", key="export_complete"):
        reporte = {
            "metadata": {
                "fecha_generacion": datetime.now().isoformat(),
                "periodo_analisis": f"{df_master['Date'].min().date()} - {df_master['Date'].max().date()}",
                "registros_totales": len(df_master),
                "estados_analizados": len(state_stats)
            },
            "resumen_ejecutivo": {
                "precio_inicial": float(precio_inicial),
                "precio_final": float(precio_final),
                "incremento_total_pct": float(incremento_total),
                "correlacion_maiz": float(corr_maiz)
            },
            "analisis_presidencial": df_pres.to_dict(orient='records'),
            "impacto_eventos": df_eventos.to_dict(orient='records')
        }
        
        path = export_results_to_json(reporte, "reporte_completo.json")
        st.success(f"✅ Reporte exportado: {path}")
        
        st.download_button(
            "⬇️ Descargar Reporte JSON",
            data=json.dumps(reporte, indent=2, ensure_ascii=False),
            file_name="reporte_completo_tortilla.json",
            mime="application/json"
        )

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p style='font-size: 1.1rem;'><b>📊 Dashboard de Análisis de Precios de Tortilla en México</b></p>
    <p>Datos: INEGI, Banco de México, CONEVAL</p>
    <p>Última actualización: {df_master['Date'].max().strftime('%d/%m/%Y')}</p>
    <p>Desarrollado con ❤️ usando Streamlit + Plotly</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        📁 Todos los resultados se exportan a: <code>{OUTPUT_DIR}/</code>
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.info(f"""
**Datos cargados:**
- {len(df_master):,} registros mensuales
- {len(state_stats)} estados
- {len(df_pres)} administraciones
- {len(df_eventos)} eventos críticos

**Periodo:** {df_master['Date'].min().year}-{df_master['Date'].max().year}
""")

st.sidebar.markdown("### 📚 Referencias")
st.sidebar.markdown("""
- **INEGI**: Precios tortilla
- **Banxico**: INPP Maíz
- **CONEVAL**: Líneas pobreza
- **STPS**: Salario mínimo
""")

st.sidebar.markdown("---")
st.sidebar.success("✅ Dashboard cargado exitosamente")