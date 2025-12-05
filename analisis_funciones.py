"""
============================================================================
FUNCIONES DE ANÁLISIS - PRECIOS DE TORTILLA EN MÉXICO
============================================================================
ARCHIVO: analisis_funciones.py

Versión optimizada con manejo consistente de fechas y código modular.
============================================================================
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import json
import re
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# =========================================================================
# CONSTANTES
# =========================================================================
DATA_DIR = "./data"
OUTPUT_DIR = "./tortilla_streamlit_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4',
    'accent': '#FFE66D',
    'purple': "#8605FF",
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'success': '#06A77D',
    'calderon': '#0051A5',
    'pena': '#FF0000',
    'amlo': '#B8860B',
    'sheinbaum': '#8B4513',
    'crisis': '#FF4444',
    'sanitario': '#FF8C00',
    'politico': '#9B59B6',
}

CONSUMO_PROMEDIO_PERSONA = 10
PERSONAS_HOGAR = 3.6
DIAS_MES_PROMEDIO = 30.4

PRESIDENTES = {
    'Felipe Calderón': {
        'partido': 'PAN',
        'inicio': '2006-12-01',
        'fin': '2012-11-30',
        'color': COLORS['calderon']
    },
    'Enrique Peña Nieto': {
        'partido': 'PRI',
        'inicio': '2012-12-01',
        'fin': '2018-11-30',
        'color': COLORS['pena']
    },
    'Andrés Manuel López Obrador': {
        'partido': 'MORENA',
        'inicio': '2018-12-01',
        'fin': '2024-09-30',
        'color': COLORS['amlo']
    },
    'Claudia Sheinbaum': {
        'partido': 'MORENA',
        'inicio': '2024-10-01',
        'fin': '2025-12-31',
        'color': COLORS['sheinbaum']
    }
}

EVENTOS_CRITICOS = [
    {'fecha': '2008-09-15', 'nombre': 'Crisis Financiera Global', 'tipo': 'economico', 'impacto': 'alto'},
    {'fecha': '2009-04-01', 'nombre': 'Pandemia H1N1', 'tipo': 'sanitario', 'impacto': 'medio'},
    {'fecha': '2014-01-01', 'nombre': 'Reforma Energética', 'tipo': 'politico', 'impacto': 'medio'},
    {'fecha': '2017-01-01', 'nombre': 'Gasolinazo Masivo', 'tipo': 'economico', 'impacto': 'alto'},
    {'fecha': '2020-05-20', 'nombre': 'COVID-19 Confinamiento', 'tipo': 'sanitario', 'impacto': 'alto'},
    {'fecha': '2022-06-01', 'nombre': 'Inflación Histórica 8%+', 'tipo': 'economico', 'impacto': 'alto'},
    {'fecha': '2024-10-01', 'nombre': 'Transición Presidencial', 'tipo': 'politico', 'impacto': 'medio'}
]

# =========================================================================
# UTILIDADES DE FECHA - MANEJO UNIFICADO
# =========================================================================

def to_datetime(date_obj) -> datetime:
    """
    Convierte cualquier tipo de fecha a datetime de Python puro.
    
    Args:
        date_obj: Objeto de fecha (datetime, Timestamp, string, etc.)
    
    Returns:
        datetime: Fecha como datetime.datetime
    """
    if pd.isna(date_obj):
        return None
    if isinstance(date_obj, datetime):
        return date_obj
    if isinstance(date_obj, pd.Timestamp):
        return date_obj.to_pydatetime()
    if isinstance(date_obj, np.datetime64):
        return pd.Timestamp(date_obj).to_pydatetime()
    if isinstance(date_obj, str):
        return pd.to_datetime(date_obj).to_pydatetime()
    return date_obj

def to_datetime_array(date_series):
    """
    Convierte una Serie de pandas o array de fechas a lista de datetime de Python.
    Útil para Plotly que no maneja bien pd.Timestamp en add_vline/add_vrect.
    
    Args:
        date_series: Series de pandas o array con fechas
    
    Returns:
        list: Lista de datetime.datetime
    """
    if isinstance(date_series, pd.Series):
        return pd.to_datetime(date_series).dt.to_pydatetime().tolist()
    elif isinstance(date_series, pd.DatetimeIndex):
        return date_series.to_pydatetime().tolist()
    else:
        return [to_datetime(d) for d in date_series]

def normalize_datetime_column(df: pd.DataFrame, col: str = 'Date') -> pd.DataFrame:
    """
    Normaliza columna de fechas a datetime de Python consistente.
    
    Args:
        df: DataFrame con columna de fechas
        col: Nombre de la columna de fecha
    
    Returns:
        DataFrame con fechas normalizadas
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def parse_mes_ano(mes_ano: str) -> Optional[datetime]:
    """Parsea formato 'ene-92' a datetime"""
    try:
        meses = {'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12}
        mes, ano = mes_ano.split('-')
        ano = int(ano)
        ano = 1900 + ano if ano > 50 else 2000 + ano
        return datetime(ano, meses[mes.lower()], 1)
    except:
        return None

def parse_trimestre(periodo: str) -> Optional[datetime]:
    """Parsea 'T1 2005' a datetime"""
    try:
        match = re.match(r'T(\d+)\s+(\d+)', periodo)
        if match:
            trimestre, ano = int(match.group(1)), int(match.group(2))
            mes = (trimestre - 1) * 3 + 1
            return datetime(ano, mes, 1)
    except:
        pass
    return None

# =========================================================================
# UTILIDADES GENERALES
# =========================================================================

def normalize_state_name(state_csv: str) -> str:
    """Normaliza nombres de estados CSV → JSON"""
    state = str(state_csv).replace("\xa0", " ").strip()
    
    mapping = {
        "D.F.": "Ciudad de México",
        "DF": "Ciudad de México",
        "Edo. México": "México",
        "Estado de México": "México",
    }
    
    return mapping.get(state, state)

def get_presidente_actual(fecha) -> Tuple[Optional[str], Optional[str]]:
    """Retorna el presidente en funciones en una fecha dada"""
    fecha_ts = pd.Timestamp(fecha)
    for nombre, info in PRESIDENTES.items():
        inicio = pd.Timestamp(info['inicio'])
        fin = pd.Timestamp(info['fin'])
        if inicio <= fecha_ts <= fin:
            return nombre, info['partido']
    return None, None

def export_results_to_json(data_dict: dict, filename: str) -> str:
    """Exporta resultados a JSON para análisis"""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False, default=str)
    return path

def export_results_to_csv(df: pd.DataFrame, filename: str) -> str:
    """Exporta DataFrame a CSV"""
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding='utf-8')
    return path

# =========================================================================
# CARGADORES DE DATOS MODULARES
# =========================================================================


MAP_PATH = "./data/mexicoHigh.json"

# ============================================================================
# 3. FUNCIÓN PARA CARGAR GEOJSON DE MÉXICO
# Agregar en analisis_funciones.py al final
# ============================================================================

def load_mexico_geojson():
    """
    Carga el GeoJSON de México y lo prepara para Plotly.
    
    Returns:
        dict: GeoJSON listo para usar en plotly.express
    """
    try:
        with open(MAP_PATH, 'r', encoding='utf-8') as f:
            geojson = json.load(f)
        
        # Normalizar nombres de estados en el GeoJSON
        for feature in geojson['features']:
            if 'properties' in feature and 'name' in feature['properties']:
                state_name = feature['properties']['name'].strip()
                # Asegurar que coincidan con los nombres del CSV
                feature['properties']['name'] = normalize_state_name(state_name)
        
        return geojson
    except FileNotFoundError:
        print(f"⚠️ Archivo GeoJSON no encontrado: {MAP_PATH}")
        return None
    except Exception as e:
        print(f"❌ Error cargando GeoJSON: {e}")
        return None

def prepare_map_data(df_tortilla, price_col="Price per kilogram"):
    """
    Prepara datos agregados por estado para visualización en mapa.
    
    Args:
        df_tortilla: DataFrame con datos de tortilla
        price_col: Nombre de columna de precio
    
    Returns:
        DataFrame con estadísticas por estado
    """
    state_data = df_tortilla.groupby('State').agg({
        price_col: ['mean', 'min', 'max', 'std', 'count'],
        'Year': ['min', 'max']
    }).round(2)
    
    state_data.columns = ['Precio_Promedio', 'Precio_Min', 'Precio_Max', 
                         'Std', 'Registros', 'Año_Inicio', 'Año_Fin']
    state_data = state_data.reset_index()
    state_data['Rango'] = state_data['Precio_Max'] - state_data['Precio_Min']
    state_data['CV'] = (state_data['Std'] / state_data['Precio_Promedio'] * 100).round(2)
    
    return state_data

def prepare_temporal_map_data(df_tortilla, year=None, price_col="Price per kilogram"):
    """
    Prepara datos por estado para un año específico o último disponible.
    
    Args:
        df_tortilla: DataFrame con datos de tortilla
        year: Año específico (None = último disponible)
        price_col: Nombre de columna de precio
    
    Returns:
        DataFrame con precios por estado para ese año
    """
    if year is None:
        year = df_tortilla['Year'].max()
    
    year_data = df_tortilla[df_tortilla['Year'] == year].copy()
    
    state_year_data = year_data.groupby('State')[price_col].mean().reset_index()
    state_year_data.columns = ['State', 'Precio']
    state_year_data['Año'] = year
    
    return state_year_data

def load_tortilla_data() -> Tuple[pd.DataFrame, bool]:
    """
    Carga datos de precios de tortilla.
    
    Returns:
        Tuple[DataFrame, bool]: (df_tortilla, HAS_CITIES)
    """
    print("📊 Cargando datos de tortilla...")
    df = pd.read_csv(f"{DATA_DIR}/tortilla_prices.csv")
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    
    price_col = "Price per kilogram"
    df["State_Original"] = df["State"]
    df["State"] = df["State"].apply(normalize_state_name)
    df = df[(df[price_col] > 0) & (df[price_col].notna())].copy()
    
    # Normalizar fechas
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
    
    HAS_CITIES = "City" in df.columns
    if HAS_CITIES:
        df["City"] = df["City"].str.strip()
    
    print(f"   ✓ {len(df):,} registros cargados")
    return df, HAS_CITIES

def load_maiz_data() -> pd.DataFrame:
    """Carga datos de INPP Maíz"""
    print("🌽 Cargando datos de maíz...")
    df = pd.read_csv(f"{DATA_DIR}/INPP_historico_maiz.csv")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Periodos'], format='%Y/%m')
    df = df.rename(columns={'INPP_SCIAN2018': 'INPP_Maiz'})
    df = df[['Date', 'INPP_Maiz']].sort_values('Date')
    print(f"   ✓ {len(df):,} registros cargados")
    return df

def load_salario_data() -> pd.DataFrame:
    """Carga datos de salario mínimo"""
    print("💵 Cargando datos de salario...")
    df = pd.read_csv(f"{DATA_DIR}/Salario_Mínimo_General Histórico_Diario_2005to2025.csv")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
    df = df.rename(columns={'Salarios Mínimos, General, Pesos por día': 'Salario_Minimo'})
    df = df[['Date', 'Salario_Minimo']].sort_values('Date')
    print(f"   ✓ {len(df):,} registros cargados")
    return df

def load_pobreza_data() -> pd.DataFrame:
    """
    Carga datos de líneas de pobreza y canasta alimentaria.
    
    Returns:
        DataFrame con columnas: Date, Canasta_Urbana, Canasta_Alimentaria, Canasta_Rural
    """
    print("📉 Cargando datos de pobreza y canasta alimentaria...")
    
    try:
        df = pd.read_csv(f"{DATA_DIR}/Valor_de_las_Líneas_de_Pobreza_por_Ingresos.csv", encoding='utf-8')
    except FileNotFoundError:
        print("   ⚠️ Archivo no encontrado, intentando con encoding alternativo...")
        df = pd.read_csv(f"{DATA_DIR}/Valor_de_las_Líneas_de_Pobreza_por_Ingresos.csv", encoding='latin-1')
    
    df.columns = df.columns.str.strip()
    
    # Encontrar columna de fecha
    fecha_col = [col for col in df.columns if 'mes' in col.lower() or 'ano' in col.lower() or 'fecha' in col.lower()]
    if fecha_col:
        df['Date'] = df[fecha_col[0]].apply(parse_mes_ano)
    else:
        # Intentar detectar formato de fecha automáticamente
        for col in df.columns:
            try:
                test_date = pd.to_datetime(df[col].iloc[0], errors='coerce')
                if pd.notna(test_date):
                    df['Date'] = pd.to_datetime(df[col])
                    break
            except:
                continue
    
    # Identificar y limpiar columnas de valores monetarios
    valor_cols = [col for col in df.columns if any(x in col.lower() for x in ['canasta', 'urbana', 'rural', 'alimentaria', 'línea', 'linea'])]
    
    for col in valor_cols:
        if df[col].dtype == 'object':
            # Limpiar símbolos monetarios y comas
            df[col] = df[col].astype(str).str.replace('$', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = df[col].str.replace(' ', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Mapeo inteligente de columnas
    new_names = {}
    
    # Prioridad para Canasta Urbana (línea de pobreza por ingresos)
    for col in df.columns:
        col_lower = col.lower()
        
        # Canasta Urbana (línea de pobreza)
        if 'urbana' in col_lower and 'canasta' not in col_lower and 'extrema' not in col_lower:
            if 'Canasta_Urbana' not in new_names.values():
                new_names[col] = 'Canasta_Urbana'
        
        # Canasta Alimentaria Urbana (línea de pobreza extrema)
        elif ('alimentaria' in col_lower or 'extrema' in col_lower) and 'urbana' in col_lower:
            if 'Canasta_Alimentaria' not in new_names.values():
                new_names[col] = 'Canasta_Alimentaria'
        
        # Canasta Rural
        elif 'rural' in col_lower and 'extrema' not in col_lower and 'alimentaria' not in col_lower:
            if 'Canasta_Rural' not in new_names.values():
                new_names[col] = 'Canasta_Rural'
    
    # Aplicar renombramientos
    df = df.rename(columns=new_names)
    
    # Eliminar columnas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Validar que tengamos la columna crítica
    if 'Canasta_Urbana' not in df.columns:
        print("   ⚠️ Advertencia: 'Canasta_Urbana' no detectada automáticamente")
        print("   📋 Columnas disponibles:", df.columns.tolist())
        
        # Intentar detectar manualmente la columna más probable
        for col in df.columns:
            if 'urbana' in col.lower() and df[col].dtype in ['float64', 'int64']:
                print(f"   🔄 Usando '{col}' como Canasta_Urbana")
                df['Canasta_Urbana'] = df[col]
                break
    
    # Seleccionar solo columnas relevantes
    keep_cols = ['Date']
    for col in ['Canasta_Urbana', 'Canasta_Alimentaria', 'Canasta_Rural']:
        if col in df.columns:
            keep_cols.append(col)
    
    df = df[keep_cols].dropna(subset=['Date']).sort_values('Date')
    
    print(f"   ✓ {len(df):,} registros cargados")
    print(f"   ✓ Columnas finales: {keep_cols}")
    print(f"   ✓ Rango de fechas: {df['Date'].min().date()} - {df['Date'].max().date()}")
    
    # Mostrar estadísticas básicas
    if 'Canasta_Urbana' in df.columns:
        print(f"   ✓ Canasta Urbana - Min: ${df['Canasta_Urbana'].min():.2f}, Max: ${df['Canasta_Urbana'].max():.2f}")
    
    return df

def load_itlp_data() -> pd.DataFrame:
    """Carga datos de ITLP (informalidad laboral)"""
    print("📊 Cargando datos de ITLP...")
    df = pd.read_csv(f"{DATA_DIR}/ITLP.csv")
    df.columns = df.columns.str.strip()
    
    id_vars = ['Estado']
    value_vars = [col for col in df.columns if col.startswith('T')]
    
    df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                      var_name='Periodo', value_name='ITLP')
    df_long['Date'] = df_long['Periodo'].apply(parse_trimestre)
    df_long = df_long.dropna(subset=['Date'])
    df_long['ITLP'] = pd.to_numeric(df_long['ITLP'], errors='coerce')
    df_long['Estado'] = df_long['Estado'].str.upper()
    
    df_nacional = df_long[df_long['Estado'] == 'NACIONAL'][['Date', 'ITLP']].sort_values('Date')
    print(f"   ✓ {len(df_nacional):,} registros nacionales cargados")
    return df_nacional

# =========================================================================
# CONSTRUCCIÓN DATASET MAESTRO
# =========================================================================

def build_master_dataset(df_tortilla: pd.DataFrame, df_maiz: pd.DataFrame, 
                        df_salario: pd.DataFrame, df_pobreza: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataset maestro con todas las métricas.
    
    Args:
        df_tortilla: DataFrame de precios de tortilla
        df_maiz: DataFrame de INPP maíz
        df_salario: DataFrame de salarios
        df_pobreza: DataFrame de líneas de pobreza y canasta alimentaria
    
    Returns:
        DataFrame maestro con todas las métricas calculadas
    """
    print("\n🔧 Construyendo dataset maestro...")
    
    # Agregación mensual de tortilla
    price_col = "Price per kilogram"
    df_tortilla['YearMonth'] = df_tortilla['Date'].dt.to_period('M')
    tortilla_mensual = df_tortilla.groupby('YearMonth')[price_col].mean().reset_index()
    tortilla_mensual['Date'] = tortilla_mensual['YearMonth'].dt.to_timestamp()
    tortilla_mensual = tortilla_mensual.rename(columns={price_col: 'Precio_Tortilla'})
    
    # Salario mensual (tomar el primer valor del mes)
    salario_mensual = df_salario.copy()
    salario_mensual['YearMonth'] = salario_mensual['Date'].dt.to_period('M')
    salario_mensual = salario_mensual.groupby('YearMonth')['Salario_Minimo'].first().reset_index()
    salario_mensual['Date'] = salario_mensual['YearMonth'].dt.to_timestamp()
    
    # Merge principal: tortilla + maíz + salario
    df_master = tortilla_mensual[['Date', 'Precio_Tortilla']].merge(
        df_maiz[['Date', 'INPP_Maiz']], on='Date', how='left'
    ).merge(
        salario_mensual[['Date', 'Salario_Minimo']], on='Date', how='left'
    )
    
    # Merge con datos de pobreza/canasta
    if df_pobreza is not None and len(df_pobreza) > 0:
        available_cols = ['Date']
        for col in ['Canasta_Urbana', 'Canasta_Alimentaria', 'Canasta_Rural']:
            if col in df_pobreza.columns:
                available_cols.append(col)
        
        if len(available_cols) > 1:
            print(f"   ✓ Integrando datos de canasta: {available_cols[1:]}")
            df_master = df_master.merge(df_pobreza[available_cols], on='Date', how='left')
            
            # Reporte de cobertura
            for col in available_cols[1:]:
                pct_cobertura = (df_master[col].notna().sum() / len(df_master)) * 100
                print(f"   ✓ {col}: {pct_cobertura:.1f}% cobertura")
    
    # Interpolación de valores faltantes
    df_master = df_master.sort_values('Date')
    numeric_cols = ['INPP_Maiz', 'Salario_Minimo']
    
    # Agregar canasta a interpolación si existe
    if 'Canasta_Urbana' in df_master.columns:
        numeric_cols.append('Canasta_Urbana')
    if 'Canasta_Alimentaria' in df_master.columns:
        numeric_cols.append('Canasta_Alimentaria')
    if 'Canasta_Rural' in df_master.columns:
        numeric_cols.append('Canasta_Rural')
    
    # Interpolación lineal para llenar huecos
    df_master[numeric_cols] = df_master[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    # Calcular métricas derivadas
    df_master = calculate_derived_metrics(df_master)
    
    # Agregar información presidencial
    df_master['Presidente'], df_master['Partido'] = zip(*df_master['Date'].apply(get_presidente_actual))
    
    # Reporte final
    print(f"   ✓ Dataset maestro creado: {len(df_master):,} registros")
    print(f"   ✓ Rango temporal: {df_master['Date'].min().date()} - {df_master['Date'].max().date()}")
    print(f"   ✓ Columnas totales: {len(df_master.columns)}")
    
    # Verificar columnas críticas
    columnas_criticas = ['Precio_Tortilla', 'INPP_Maiz', 'Salario_Minimo', 'Canasta_Urbana']
    for col in columnas_criticas:
        if col in df_master.columns:
            valores_validos = df_master[col].notna().sum()
            pct = (valores_validos / len(df_master)) * 100
            print(f"   ✓ {col}: {valores_validos}/{len(df_master)} valores ({pct:.1f}%)")
    
    return df_master

def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas derivadas para el dataset maestro.
    
    Args:
        df: DataFrame con datos base
    
    Returns:
        DataFrame con métricas calculadas
    """
    df = df.copy()
    
    # Métricas básicas de poder adquisitivo
    df['Salario_Mensual'] = df['Salario_Minimo'] * DIAS_MES_PROMEDIO
    df['Kg_Tortilla_x_Salario'] = df['Salario_Mensual'] / df['Precio_Tortilla']
    df['Consumo_Hogar_kg'] = CONSUMO_PROMEDIO_PERSONA * PERSONAS_HOGAR
    df['Gasto_Tortilla_Hogar'] = df['Consumo_Hogar_kg'] * df['Precio_Tortilla']
    df['Pct_Salario_en_Tortilla'] = (df['Gasto_Tortilla_Hogar'] / df['Salario_Mensual']) * 100
    
    # Índices base 100 (año 2007)
    base_year_data = df[df['Date'].dt.year == 2007]
    
    if len(base_year_data) > 0:
        base_date = base_year_data['Date'].min()
        base_row = df[df['Date'] == base_date].iloc[0]
        
        # Índices principales
        df['Indice_Tortilla'] = (df['Precio_Tortilla'] / base_row['Precio_Tortilla']) * 100
        df['Indice_Maiz'] = (df['INPP_Maiz'] / base_row['INPP_Maiz']) * 100
        df['Indice_Salario'] = (df['Salario_Minimo'] / base_row['Salario_Minimo']) * 100
        df['Indice_Accesibilidad'] = (df['Indice_Salario'] / df['Indice_Tortilla']) * 100
        
        # Índice de Canasta (si existe)
        if 'Canasta_Urbana' in df.columns:
            # Verificar que el valor base no sea NaN
            if pd.notna(base_row.get('Canasta_Urbana', np.nan)):
                df['Indice_Canasta'] = (df['Canasta_Urbana'] / base_row['Canasta_Urbana']) * 100
            else:
                # Si no hay dato en 2007, usar el primer valor disponible
                first_valid_canasta = df['Canasta_Urbana'].first_valid_index()
                if first_valid_canasta is not None:
                    base_canasta = df.loc[first_valid_canasta, 'Canasta_Urbana']
                    df['Indice_Canasta'] = (df['Canasta_Urbana'] / base_canasta) * 100
                    print(f"   ⚠️ Base de Canasta ajustada a {df.loc[first_valid_canasta, 'Date'].date()}")
        
        # Métricas adicionales de canasta
        if 'Canasta_Alimentaria' in df.columns:
            if pd.notna(base_row.get('Canasta_Alimentaria', np.nan)):
                df['Indice_Canasta_Alimentaria'] = (df['Canasta_Alimentaria'] / base_row['Canasta_Alimentaria']) * 100
    else:
        print("   ⚠️ No hay datos de 2007 para calcular índices base")
    
    return df

# =========================================================================
# FUNCIÓN PRINCIPAL DE CARGA
# =========================================================================

def load_all_data() -> Tuple:
    """
    Carga todos los datasets y construye el dataset maestro.
    
    Returns:
        Tuple: (df_tortilla, df_maiz, df_salario, df_pobreza, df_itlp_nacional, df_master, HAS_CITIES)
    """
    print("\n" + "="*60)
    print("🌮 INICIANDO CARGA DE DATOS")
    print("="*60)
    
    try:
        # Cargar datasets individuales
        df_tortilla, HAS_CITIES = load_tortilla_data()
        df_maiz = load_maiz_data()
        df_salario = load_salario_data()
        df_pobreza = load_pobreza_data()
        df_itlp_nacional = load_itlp_data()
        
        # Construir dataset maestro
        df_master = build_master_dataset(df_tortilla, df_maiz, df_salario, df_pobreza)
        
        print("\n" + "="*60)
        print("✅ CARGA COMPLETADA EXITOSAMENTE")
        print("="*60 + "\n")
        
        return df_tortilla, df_maiz, df_salario, df_pobreza, df_itlp_nacional, df_master, HAS_CITIES
        
    except Exception as e:
        print(f"\n❌ ERROR durante la carga: {str(e)}")
        raise

# =========================================================================
# ANÁLISIS PRESIDENCIAL
# =========================================================================

def calcular_analisis_presidencial(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas por presidente.
    
    Args:
        df_master: DataFrame maestro con datos mensuales
    
    Returns:
        DataFrame con análisis por presidente
    """
    resultados_pres = []
    
    for nombre, info in PRESIDENTES.items():
        inicio = pd.Timestamp(info['inicio'])
        fin = pd.Timestamp(info['fin'])
        
        mask = (df_master['Date'] >= inicio) & (df_master['Date'] <= fin)
        datos_periodo = df_master[mask]
        
        if len(datos_periodo) > 0:
            precio_inicial = datos_periodo['Precio_Tortilla'].iloc[0]
            precio_final = datos_periodo['Precio_Tortilla'].iloc[-1]
            incremento_abs = precio_final - precio_inicial
            incremento_pct = (incremento_abs / precio_inicial) * 100
            
            dias = (fin - inicio).days
            anos = dias / 365.25
            
            cagr = (((precio_final / precio_inicial) ** (1 / anos)) - 1) * 100 if anos > 0 else 0
            volatilidad = datos_periodo['Precio_Tortilla'].pct_change().std() * np.sqrt(12) * 100
            
            resultados_pres.append({
                'Presidente': nombre,
                'Partido': info['partido'],
                'Precio_Inicial': precio_inicial,
                'Precio_Final': precio_final,
                'Incremento_Absoluto': incremento_abs,
                'Incremento_Porcentual': incremento_pct,
                'CAGR': cagr,
                'Volatilidad': volatilidad,
                'Promedio': datos_periodo['Precio_Tortilla'].mean(),
                'Años': anos,
                'Color': info['color']
            })
    
    return pd.DataFrame(resultados_pres)

# =========================================================================
# ANÁLISIS DE EVENTOS
# =========================================================================

def calcular_impacto_eventos(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula impacto de eventos críticos.
    
    Args:
        df_master: DataFrame maestro con datos mensuales
    
    Returns:
        DataFrame con impacto de eventos
    """
    monthly = df_master.set_index('Date')['Precio_Tortilla']
    resultados_eventos = []
    
    for evento in EVENTOS_CRITICOS:
        fecha = pd.Timestamp(evento['fecha'])
        
        inicio_antes = fecha - pd.DateOffset(months=6)
        inicio_despues = fecha + pd.DateOffset(months=1)
        fin_despues = fecha + pd.DateOffset(months=6)
        
        datos_antes = monthly[(monthly.index >= inicio_antes) & (monthly.index < fecha)]
        datos_despues = monthly[(monthly.index > inicio_despues) & (monthly.index <= fin_despues)]
        
        if len(datos_antes) > 0 and len(datos_despues) > 0:
            promedio_antes = datos_antes.mean()
            promedio_despues = datos_despues.mean()
            cambio = promedio_despues - promedio_antes
            cambio_pct = (cambio / promedio_antes) * 100
            
            resultados_eventos.append({
                'Evento': evento['nombre'],
                'Fecha': fecha.date(),
                'Tipo': evento['tipo'],
                'Impacto': evento['impacto'],
                'Precio_Antes': promedio_antes,
                'Precio_Despues': promedio_despues,
                'Cambio_Absoluto': cambio,
                'Cambio_Porcentual': cambio_pct
            })
    
    return pd.DataFrame(resultados_eventos)

# =========================================================================
# ANÁLISIS ESTATAL Y DE CIUDADES
# =========================================================================

def calcular_estadisticas_estatales(df_tortilla: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas por estado.
    
    Args:
        df_tortilla: DataFrame con datos de tortilla por estado
    
    Returns:
        DataFrame con estadísticas estatales
    """
    price_col = "Price per kilogram"
    
    state_stats = df_tortilla.groupby('State')[price_col].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).round(2)
    state_stats['range'] = state_stats['max'] - state_stats['min']
    state_stats['cv'] = (state_stats['std'] / state_stats['mean'] * 100).round(2)
    state_stats = state_stats.sort_values('mean', ascending=False)
    
    return state_stats

def calcular_estadisticas_ciudades(df_tortilla: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas por ciudad"""
    price_col = "Price per kilogram"
    
    city_counts = df_tortilla["City"].value_counts()
    valid_cities = city_counts[city_counts >= 10].index
    df_cities = df_tortilla[df_tortilla["City"].isin(valid_cities)].copy()
    
    city_stats = df_cities.groupby("City").agg({
        price_col: ['mean', 'std', 'count'],
        'State': 'first'
    }).round(2)
    city_stats.columns = ['mean', 'std', 'count', 'Estado']
    city_stats = city_stats.sort_values('mean', ascending=False)
    
    return city_stats

# =========================================================================
# PROYECCIONES
# =========================================================================

def calcular_proyecciones(df_master: pd.DataFrame) -> Dict:
    """
    Calcula proyecciones a 12 meses con múltiples modelos.
    
    Args:
        df_master: DataFrame maestro con datos mensuales
    
    Returns:
        dict con proyecciones y métricas de modelos
    """
    # Ventanas temporales
    df_recent = df_master.tail(24).copy()
    df_recent['Days_Recent'] = (df_recent['Date'] - df_recent['Date'].min()).dt.days
    
    df_immediate = df_master.tail(12).copy()
    df_immediate['Days_Immediate'] = (df_immediate['Date'] - df_immediate['Date'].min()).dt.days
    
    # Modelos
    X_24m = df_recent[['Days_Recent']].values
    y_24m = df_recent['Precio_Tortilla'].values
    model_24m = LinearRegression()
    model_24m.fit(X_24m, y_24m)
    
    X_12m = df_immediate[['Days_Immediate']].values
    y_12m = df_immediate['Precio_Tortilla'].values
    model_12m = LinearRegression()
    model_12m.fit(X_12m, y_12m)
    
    # Proyecciones
    last_date = df_master['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    
    future_days_24m = np.arange(
        df_recent['Days_Recent'].max() + 30, 
        df_recent['Days_Recent'].max() + 30 + (30 * 12), 
        30
    ).reshape(-1, 1)
    y_pred_24m = model_24m.predict(future_days_24m)
    
    future_days_12m = np.arange(
        df_immediate['Days_Immediate'].max() + 30, 
        df_immediate['Days_Immediate'].max() + 30 + (30 * 12), 
        30
    ).reshape(-1, 1)
    y_pred_12m = model_12m.predict(future_days_12m)
    
    # Conservadora
    rolling_12m = df_master['Precio_Tortilla'].tail(12).mean()
    inflation_rate = 0.04 / 12
    y_pred_conservative = np.array([rolling_12m * (1 + inflation_rate) ** i for i in range(1, 13)])
    
    # Banda de confianza
    std_12m = df_master.tail(12)['Precio_Tortilla'].std()
    
    # R² y pendientes
    r2_24m = model_24m.score(X_24m, y_24m)
    slope_24m = model_24m.coef_[0] * 365.25
    
    r2_12m = model_12m.score(X_12m, y_12m)
    slope_12m = model_12m.coef_[0] * 365.25
    
    return {
        'future_dates': future_dates,
        'y_pred_24m': y_pred_24m.flatten(),
        'y_pred_12m': y_pred_12m.flatten(),
        'y_pred_conservative': y_pred_conservative,
        'std_12m': std_12m,
        'r2_24m': r2_24m,
        'slope_24m': slope_24m,
        'r2_12m': r2_12m,
        'slope_12m': slope_12m,
        'model_24m': model_24m,
        'model_12m': model_12m
    }