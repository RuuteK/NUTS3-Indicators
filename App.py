import json
import os
import socket
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from branca.colormap import linear

# --- Obsługa zasobów w EXE (PyInstaller) ---
import sys


def resource_path(filename: str) -> Path:
    """Ścieżka do zasobu tak w .py, jak i w 'onefile' .exe (PyInstaller)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / filename


def prefer_external(filename: str) -> Path:
    """Jeśli obok EXE leży zewnętrzny plik – użyj jego; inaczej zasób wbudowany."""
    p = Path.cwd() / filename
    return p if p.exists() else resource_path(filename)


# ==== USTAW ŚCIEŻKI (pliki w tym samym folderze co app.py) ====
DATA_PATH = prefer_external("OE NUTS3 Europe_reshaped_long_for_arcgis.xlsx")  # CSV lub XLSX w układzie długim
GEO_PATH = prefer_external("Data_Europe_NUTS3_new.geojson")  # GeoJSON NUTS3 (4326, uproszczony)

st.set_page_config(page_title="NUTS3 Indicators", layout="wide")


# -------------------- Podkłady --------------------
BASEMAPS: Dict[str, object] = {
    "Carto Positron": "cartodbpositron",
    "Carto DarkMatter": "cartodbdark_matter",
    "OpenStreetMap": "OpenStreetMap",
    "Stamen Terrain": "Stamen Terrain",
    "Stamen Toner": "Stamen Toner",
    "Esri WorldStreetMap": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "Tiles © Esri — Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors",
    ),
    "Esri WorldImagery": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Tiles © Esri — Esri & contributors",
    ),
    "Esri Light Gray": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "Tiles © Esri — Esri, DeLorme, NAVTEQ",
    ),
}

# Kody krajów -> nazwy (PL)
COUNTRY_NAMES_PL = {
    "AT": "Austria",
    "BE": "Belgia",
    "BG": "Bułgaria",
    "CH": "Szwajcaria",
    "CY": "Cypr",
    "CZ": "Czechy",
    "DE": "Niemcy",
    "DK": "Dania",
    "EE": "Estonia",
    "EL": "Grecja",
    "GR": "Grecja",
    "ES": "Hiszpania",
    "FI": "Finlandia",
    "FR": "Francja",
    "HR": "Chorwacja",
    "HU": "Węgry",
    "IE": "Irlandia",
    "IS": "Islandia",
    "IT": "Włochy",
    "LT": "Litwa",
    "LU": "Luksemburg",
    "LV": "Łotwa",
    "MT": "Malta",
    "NL": "Holandia",
    "NO": "Norwegia",
    "PL": "Polska",
    "PT": "Portugalia",
    "RO": "Rumunia",
    "SE": "Szwecja",
    "SI": "Słowenia",
    "SK": "Słowacja",
    "UK": "Wielka Brytania",
    "GB": "Wielka Brytania",
    "ME": "Czarnogóra",
    "RS": "Serbia",
    "AL": "Albania",
    "BA": "Bośnia i Hercegowina",
    "MK": "Macedonia Północna",
    "UA": "Ukraina",
    "RU": "Rosja",
    "TR": "Turcja",
}


def code_to_name(code: str) -> str:
    return COUNTRY_NAMES_PL.get(str(code).upper(), str(code))


# -------------------- Helpery --------------------
def _clean(s: str) -> str:
    return s.strip().replace("\u00A0", " ").replace("  ", " ").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    synonyms = {
        "NUTS_ID": ["nuts_id", "nuts 3", "nuts3", "nuts_code", "nuts_id code"],
        "CNTR_CODE": ["cntr_code", "cntr", "cntrcode", "country", "kraj"],
        "Location": ["location", "nuts_name", "name_latn", "name", "nazwa"],
        "Indicator": ["indicator", "indykator", "wskaźnik", "wskaznik"],
        "Year": ["year", "rok"],
        "Value": ["value", "wartość", "wartosc", "val"],
    }
    lower_map = {_clean(c): c for c in df.columns}
    rename_map = {}
    for std, alts in synonyms.items():
        for a in alts:
            if a in lower_map:
                rename_map[lower_map[a]] = std
                break
    df = df.rename(columns=rename_map)

    if "CNTR_CODE" not in df.columns and "NUTS_ID" in df.columns:
        df["CNTR_CODE"] = df["NUTS_ID"].astype(str).str[:2]

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    if "Value" in df.columns:
        df["Value"] = (
            df["Value"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"nan": None, "": None})
        )
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    if "NUTS_ID" in df.columns:
        df["NUTS_ID"] = df["NUTS_ID"].astype(str).str.strip()
    if "CNTR_CODE" in df.columns:
        df["CNTR_CODE"] = df["CNTR_CODE"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Brak pliku danych: {path.resolve()}")
        st.stop()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=0)
    return normalize_columns(df)


@st.cache_data(show_spinner=False)
def load_geo(path: Path):
    if not path.exists():
        st.error(f"Brak GeoJSON: {path.resolve()}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        geo = json.load(f)
    feats = geo.get("features", [])
    if not any(ft.get("geometry") for ft in feats):
        st.error(
            "GeoJSON nie zawiera geometrii (geometry=null). Wyeksportuj z SHP do GeoJSON (EPSG:4326)."
        )
        st.stop()
    for ft in feats:
        props = ft.get("properties", {}) or {}
        nuts = str(
            props.get("NUTS_ID")
            or props.get("nuts_id")
            or props.get("NUTS3")
            or ""
        ).strip()
        name = (
            props.get("NUTS_NAME")
            or props.get("NAME_LATN")
            or props.get("NAME")
            or props.get("name")
            or ""
        )
        cntr = (
            props.get("CNTR_CODE")
            or props.get("CNTRCODE")
            or props.get("CNTR")
            or props.get("cntr_code")
            or ""
        )
        if not cntr and nuts:
            cntr = nuts[:2]
        props["NUTS_ID"] = nuts
        props["NUTS_NAME"] = name
        props["CNTR_CODE"] = str(cntr)
        props["COUNTRY_NAME"] = code_to_name(cntr)
        ft["properties"] = props
    return geo


# --- Pomocnicze: wyciąganie współrzędnych i bounding box kraju ---
def _extract_coords_from_geometry(geom: dict) -> List[Tuple[float, float]]:
    """Zwraca listę (lon, lat) z dowolnej geometrii GeoJSON (Polygon/MultiPolygon itd.)."""
    if not geom:
        return []
    coords = geom.get("coordinates")
    out: List[Tuple[float, float]] = []

    def _walk(c):
        if not c:
            return
        if isinstance(c[0], (float, int)):  # pojedynczy punkt [lon, lat]
            if len(c) >= 2:
                out.append((float(c[0]), float(c[1])))
        else:
            for el in c:
                _walk(el)

    _walk(coords)
    return out


def get_country_bounds(geo: dict, country_code: str) -> Optional[List[List[float]]]:
    """Oblicza bounding box [ [min_lat,min_lon], [max_lat,max_lon] ] dla danego kraju."""
    all_coords: List[Tuple[float, float]] = []
    for ft in geo.get("features", []):
        props = ft.get("properties", {}) or {}
        code = props.get("CNTR_CODE") or props.get("CNTR") or ""
        if str(code).strip() != str(country_code).strip():
            continue
        geom = ft.get("geometry")
        all_coords.extend(_extract_coords_from_geometry(geom))

    if not all_coords:
        return None

    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    return [[min_lat, min_lon], [max_lat, max_lon]]


def build_map(
    geo_fc,
    base_choice,
    indicator,
    year,
    fit_bounds: bool = True,
    bounds: Optional[List[List[float]]] = None,
) -> folium.Map:
    """Tworzy NOWY obiekt mapy – z opcją przybliżenia do wskazanych bounds."""
    vals = [
        f["properties"].get("Value")
        for f in geo_fc["features"]
        if f["properties"].get("Value") is not None
    ]
    if len(vals) >= 2:
        s = sorted(vals)
        vmin = s[int(0.05 * len(s))]
        vmax = s[int(0.95 * len(s)) - 1]
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
    elif len(vals) == 1:
        vmin, vmax = vals[0] - 1, vals[0] + 1
    else:
        vmin, vmax = 0, 1
    cmap = linear.Blues_09.scale(vmin, vmax)

    def style_fn(feat):
        v = feat["properties"].get("Value")
        if v is None:
            return {
                "fillColor": "#dddddd",
                "color": "#666",
                "weight": 0.6,
                "fillOpacity": 0.45,
            }
        return {
            "fillColor": cmap(v),
            "color": "#666",
            "weight": 0.6,
            "fillOpacity": 0.8,
        }

    # Startowa pozycja – Europa; później nadpisujemy fit_bounds
    m = folium.Map(
        location=[50, 15],
        zoom_start=4,
        tiles=None,
        width="100%",
        height="700px",
        control_scale=True,
    )

    base = BASEMAPS[base_choice]
    if isinstance(base, str):
        folium.TileLayer(base, name=base_choice, control=False).add_to(m)
    else:
        url, attr = base
        folium.TileLayer(
            tiles=url, attr=attr, name=base_choice, control=False
        ).add_to(m)

    layer = folium.GeoJson(
        data=geo_fc,
        style_function=style_fn,
        highlight_function=lambda x: {
            "weight": 2,
            "color": "#000",
            "fillOpacity": 0.9,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["COUNTRY_NAME", "NUTS_ID", "NUTS_NAME", "Value"],
            aliases=["Kraj:", "NUTS_ID:", "Region:", "Value:"],
            sticky=False,
        ),
        name="NUTS3",
    ).add_to(m)

    cmap.caption = f"{indicator} ({year})"
    cmap.add_to(m)

    # Priorytet: jeśli podane bounds (np. jednego kraju) – przybliż do niego
    if bounds is not None:
        try:
            m.fit_bounds(bounds)
        except Exception:
            pass
    elif fit_bounds:
        try:
            m.fit_bounds(layer.get_bounds())
        except Exception:
            pass

    return m


def folium_to_html_bytes(m: folium.Map) -> bytes:
    """Zapis mapy do HTML (bytes) przez plik tymczasowy — 1 mapa w pliku."""
    fd, path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        m.save(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def folium_to_image_bytes(
    m: folium.Map, width=1400, height=900
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Eksport do PNG przy pomocy Selenium (headless Chrome/Edge).
    Zwraca (bytes, error_msg). Jeśli nie ma selenium/chromedrivera – error_msg != None.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        import time
    except Exception:
        return (
            None,
            (
                "Brak zależności do eksportu PNG. Zainstaluj:\n"
                "  pip install selenium webdriver-manager pillow\n"
                "oraz Chrome/Edge na komputerze."
            ),
        )

    fd, path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        m.save(path)

        options = Options()
        # nowe headless (Chrome 109+)
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--window-size={width},{height}")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("file:///" + path.replace("\\", "/"))
        time.sleep(2.0)  # chwila na dogranie kafli

        png = driver.get_screenshot_as_png()
        driver.quit()

        return png, None
    except Exception as e:
        return None, f"Nie udało się wykonać zrzutu: {e}"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# -------------------- Wczytanie danych --------------------
df = load_table(DATA_PATH)
geo = load_geo(GEO_PATH)

required = ["NUTS_ID", "Indicator", "Year", "Value"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Brak kolumn: {missing}. Obecne: {list(df.columns)}")
    st.stop()

# słownik NUTS_ID -> nazwa regionu
nuts_to_name = {
    ft["properties"].get("NUTS_ID", ""): ft["properties"].get("NUTS_NAME", "")
    for ft in geo.get("features", [])
}

# -------------------- Filtry NA GŁÓWNEJ STRONIE --------------------
st.markdown("### Filtry")

# przygotowanie list pomocniczych
if "CNTR_CODE" in df.columns:
    country_codes = sorted(df["CNTR_CODE"].dropna().astype(str).unique())
else:
    country_codes = []

country_names = [code_to_name(c) for c in country_codes]
name_to_code = {code_to_name(c): c for c in country_codes}

years = sorted(df["Year"].dropna().unique())
if len(years) > 0:
    if 2024 in years:
        idx_year = list(years).index(2024)
    else:
        idx_year = len(years) - 1
else:
    idx_year = 0

indicators = sorted(df["Indicator"].dropna().unique())

# układ w kolumnach nad mapą
col1, col2, col3, col4 = st.columns([1.2, 1.8, 1, 1])

with col1:
    base_choice = st.selectbox("Mapa bazowa", list(BASEMAPS.keys()), index=0)

with col2:
    sel_country_names = st.multiselect("Kraj", country_names, default=[])

with col3:
    sel_year = st.selectbox("Rok", years, index=idx_year)

with col4:
    sel_indicator = st.selectbox("Indicator", indicators)

# konwersja wybranych krajów do kodów
sel_countries = [name_to_code[n] for n in sel_country_names]

# -------------------- Filtrowanie i budowa geo --------------------
mask = (df["Year"].eq(sel_year)) & (df["Indicator"].eq(sel_indicator))
if sel_countries and "CNTR_CODE" in df.columns:
    mask &= df["CNTR_CODE"].isin(sel_countries)

cols = ["NUTS_ID", "Value", "Year", "Indicator"]
if "CNTR_CODE" in df.columns:
    cols.append("CNTR_CODE")
df_sel = df.loc[mask, cols].copy()

df_plot = df_sel.groupby("NUTS_ID", as_index=False, dropna=False)["Value"].mean()
df_plot["Region"] = df_plot["NUTS_ID"].map(nuts_to_name)

# mapy do właściwości
val_map = dict(zip(df_plot["NUTS_ID"], df_plot["Value"]))
name_map = dict(zip(df_plot["NUTS_ID"], df_plot["Region"]))

# filtruj geojson do wybranych krajów i dopisz Value/Name + Country
features = []
for ft in geo.get("features", []):
    props = ft["properties"].copy()
    nid = props.get("NUTS_ID", "")
    code = props.get("CNTR_CODE") or (nid[:2] if nid else "")
    if sel_countries and code not in sel_countries:
        continue
    props["Value"] = val_map.get(nid)
    props["NUTS_NAME"] = name_map.get(nid, props.get("NUTS_NAME", ""))
    props["COUNTRY_NAME"] = code_to_name(code)
    features.append(
        {"type": "Feature", "geometry": ft["geometry"], "properties": props}
    )

geo_filtered = {"type": "FeatureCollection", "features": features}

# bounds do przybliżenia – jeśli wybrano dokładnie 1 kraj
country_bounds = None
if len(sel_countries) == 1:
    country_bounds = get_country_bounds(geo, sel_countries[0])

# -------------------- Mapa + eksport --------------------
st.subheader(f"Mapa: {sel_indicator} — {sel_year}")
if not features:
    st.info("Brak danych/obiektów dla wybranych filtrów.")
else:
    # 1) wyświetlenie
    m_display = build_map(
        geo_filtered,
        base_choice,
        sel_indicator,
        sel_year,
        fit_bounds=(country_bounds is None),
        bounds=country_bounds,
    )
    st_folium(
        m_display,
        height=700,
        width=None,
        key=f"map_{sel_indicator}_{sel_year}_{'_'.join(sel_countries) or 'all'}",
    )

    # 2) eksport – osobny obiekt (żeby nie dublować mapy)
    m_export = build_map(
        geo_filtered,
        base_choice,
        sel_indicator,
        sel_year,
        fit_bounds=(country_bounds is None),
        bounds=country_bounds,
    )

    html_bytes = folium_to_html_bytes(m_export)
    fname_base = f"map_{sel_indicator}_{sel_year}" + (
        f"_{'_'.join(sel_countries)}" if sel_countries else ""
    )
    
    if RUNNING_IN_CLOUD:
        # W chmurze: tylko HTML, bez próby użycia Selenium
        col_html, _ = st.columns(2)
    
        with col_html:
            st.download_button(
                "Pobierz mapę (HTML)",
                data=html_bytes,
                file_name=f"{fname_base}.html",
                mime="text/html",
            )
    
        st.info("Eksport do PNG jest dostępny tylko w wersji desktopowej aplikacji.")
    else:
        # Lokalnie: HTML + PNG
        col_html, col_png = st.columns(2)
    
        with col_html:
            st.download_button(
                "Pobierz mapę (HTML)",
                data=html_bytes,
                file_name=f"{fname_base}.html",
                mime="text/html",
            )
    
        img_png, err = folium_to_image_bytes(m_export, width=1400, height=900)
    
        with col_png:
            if err:
                st.info(err)
            else:
                st.download_button(
                    "Pobierz mapę (PNG)",
                    data=img_png,
                    file_name=f"{fname_base}.png",
                    mime="image/png",
                )


# -------------------- Ranking + XLSX --------------------
st.subheader(f"Ranking regionów NUTS3 wg wartości wskaźnika: {sel_indicator} — {sel_year}")
if not df_plot.empty:
    # do tabeli i eksportu – kraj po nazwie
    df_plot["Country"] = df_plot["NUTS_ID"].str[:2].map(code_to_name)
    rank = (
        df_plot.dropna(subset=["Value"])
        .sort_values("Value", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        rank[["Country", "NUTS_ID", "Region", "Value"]],
        use_container_width=True,
    )
else:
    rank = pd.DataFrame(
        columns=["Country", "NUTS_ID", "Region", "Value"]
    )
    st.write("—")

# XLSX: arkusze 'data' i 'ranking'
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    out_data = df_sel.copy()
    if "CNTR_CODE" in out_data.columns:
        out_data["Country"] = out_data["CNTR_CODE"].map(code_to_name)
        cols_order = ["Country", "CNTR_CODE"] + [
            c
            for c in out_data.columns
            if c not in ("Country", "CNTR_CODE")
        ]
        out_data = out_data[cols_order]
    out_data.to_excel(writer, index=False, sheet_name="data")

    out_rank = rank[["Country", "NUTS_ID", "Region", "Value"]].copy()
    out_rank.to_excel(writer, index=False, sheet_name="ranking")

buffer.seek(0)
st.download_button(
    "Pobierz dane (XLSX)",
    data=buffer.getvalue(),
    file_name=(
        f"nuts3_{sel_indicator}_{sel_year}"
        + (f"_{'_'.join(sel_countries)}" if sel_countries else "")
        + ".xlsx"
    ),
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

