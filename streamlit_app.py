"""
Garden Planner — Streamlit App (Updated)
Adds: Visual Garden Planner tab, cluster companion badges,
      downloadable standalone HTML planner pre-seeded with results.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import traceback

sys.path.insert(0, str(Path(__file__).parent))

try:
    from garden_planner_core import GardenPlanner, PlantClusteringModule, Config
    from climate_projection import get_climate_projection_for_location
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.error("Please ensure garden_planner_core.py and climate_projection.py are in the same directory.")
    st.stop()

st.set_page_config(
    page_title="🌱 Garden Planner",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #2E7D32; text-align: center; margin-bottom: 1rem; font-weight: bold; }
    .subtitle { text-align: center; color: #558B2F; font-size: 1.2rem; margin-bottom: 2rem; }
    .plant-card { background-color: #F1F8F4; padding: 1rem; border-radius: 10px; border-left: 4px solid #2E7D32; margin: 0.5rem 0; }
    .metric-container { background-color: #E8F5E9; padding: 0.5rem; border-radius: 5px; text-align: center; }
    .info-badge { background-color: #4CAF50; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.85rem; font-weight: bold; }
    .download-section { background-color: #F5F5F5; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
    .climate-warning { background-color: #FFF3E0; border-left: 4px solid #FF9800; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .climate-info { background-color: #E3F2FD; border-left: 4px solid #2196F3; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
    .cluster-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem; font-weight: bold; margin-right: 4px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌱 Garden Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent plant recommendations with climate change projections</p>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Garden Configuration")

EXAMPLE_LOCATIONS = {
    "🇧🇬 Bulgaria": [
        ("Sofia",         42.6977,  23.3219),
        ("Plovdiv",       42.1369,  24.7827),
        ("Varna",         43.2100,  27.9361),
        ("Burgas",        42.5048,  27.4626),
        ("Teteven",       42.9197,  24.2664),
        ("Vladaya",       42.6200,  23.2300),
    ],
    "🌍 World": [
        ("London",        51.5074,  -0.1278),
        ("New York",      40.7128, -74.0060),
        ("Paris",         48.8566,   2.3522),
        ("Tokyo",         35.6762, 139.6503),
        ("Buenos Aires", -34.6067, -58.4362),
    ],
}

# Initialise session state keys for lat/lon so buttons can overwrite them
if "latitude" not in st.session_state:
    st.session_state["latitude"] = 42.6977
if "longitude" not in st.session_state:
    st.session_state["longitude"] = 23.3219
if "garden_name" not in st.session_state:
    st.session_state["garden_name"] = "My Garden"

with st.sidebar:
    st.markdown("### 📍 Location")
    garden_name = st.text_input("Garden Name", key="garden_name")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input(
            "Latitude", key="latitude",
            format="%.4f", min_value=-90.0, max_value=90.0
        )
    with col2:
        longitude = st.number_input(
            "Longitude", key="longitude",
            format="%.4f", min_value=-180.0, max_value=180.0
        )
    st.info("💡 **Tip:** Right-click on Google Maps and copy coordinates")

    with st.expander("📍 Example locations — click to use", expanded=False):
        for region, cities in EXAMPLE_LOCATIONS.items():
            st.markdown(f"**{region}**")
            for city_name, lat, lon in cities:
                col_btn, col_coords = st.columns([3, 2])
                with col_btn:
                    if st.button(city_name, key=f"loc_{city_name}", use_container_width=True):
                        st.session_state["latitude"] = lat
                        st.session_state["longitude"] = lon
                        st.session_state["garden_name"] = city_name
                        st.rerun()
                with col_coords:
                    st.caption(f"`{lat}, {lon}`")

    st.markdown("---")
    st.markdown("### 🎛️ Preferences")
    num_rec = st.slider("Number of Plants", 10, 100, 30, 10)
    min_score = st.slider("Minimum Suitability", 0.0, 1.0, 0.5, 0.05)
    max_cluster = st.slider("Plants per Cluster", 3, 10, 5, 1)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        generate = st.button("🌿 Generate", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ── Field explanations ────────────────────────────────────────────────────────

FIELD_EXPLANATIONS = {
    'Shade': {
        'F': 'Full Sun - Needs direct sunlight most of the day',
        'S': 'Semi-Shade - Tolerates partial shade',
        'N': 'Full Shade - Thrives in shaded conditions'
    },
    'Moisture': {
        'D': 'Dry - Prefers well-drained, dry soil',
        'M': 'Moist - Needs consistently moist soil',
        'We': 'Wet - Tolerates waterlogged conditions',
        'Wa': 'Water - Aquatic plant, grows in water'
    },
    'Soil': {
        'L': 'Light - Sandy, well-draining soil',
        'M': 'Medium - Loamy soil',
        'H': 'Heavy - Clay soil',
        'acid': 'Acidic - pH < 6.5',
        'neutral': 'Neutral - pH 6.5-7.5',
        'alkaline': 'Alkaline - pH > 7.5'
    }
}

CLUSTER_COLORS = [
    "#4A7C59", "#7B5EA7", "#C07020", "#2271B3", "#C0392B",
    "#1A7A6A", "#8B6340", "#5C7A2A", "#B03060", "#2C5F8A"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_legend_to_image(image_path):
    try:
        img = Image.open(image_path)
        legend_height = 150
        new_img = Image.new('RGB', (img.width, img.height + legend_height), 'white')
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        legend_y = img.height + 20
        draw.text((20, legend_y), "Plant Clustering Visualization", font=title_font, fill='#2E7D32')
        legend_y += 30
        draw.text((20, legend_y),
            "• Each color represents a cluster of plants with similar growing requirements\n"
            "• Plants in the same cluster grow well together (companion planting)\n"
            "• Distance between points shows how similar plants are",
            font=text_font, fill='#333333')
        img_bytes = io.BytesIO()
        new_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    except Exception:
        with open(image_path, 'rb') as f:
            return io.BytesIO(f.read())


def cluster_color(cluster_id):
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


def pick_emoji(name, habit="", edibility=""):
    n = str(name).lower()
    mapping = {
        'tomato': '🍅', 'rose': '🌹', 'sunflower': '🌻', 'carrot': '🥕',
        'lettuce': '🥬', 'salad': '🥬', 'strawberr': '🍓', 'basil': '🌿',
        'mint': '🌿', 'lavender': '💜', 'pea': '🫛', 'potato': '🥔',
        'courgette': '🥒', 'zucchini': '🥒', 'cucumber': '🥒', 'apple': '🍎',
        'pear': '🍐', 'cherry': '🍒', 'plum': '🫐', 'grape': '🍇',
        'bean': '🫘', 'onion': '🧅', 'garlic': '🧄', 'corn': '🌽',
        'pepper': '🌶', 'eggplant': '🍆', 'aubergine': '🍆',
    }
    for key, emoji in mapping.items():
        if key in n:
            return emoji
    h = str(habit).lower()
    if 'tree' in h:
        return '🌳'
    if 'shrub' in h:
        return '🌲'
    try:
        if int(edibility) >= 3:
            return '🥦'
    except Exception:
        pass
    return '🌱'


def build_planner_html(df, garden_name, impact_level=""):
    """Build a self-contained HTML garden planner pre-loaded with plant data."""

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    def estimate_sow(hardiness, shade):
        try:
            h = int(hardiness)
        except Exception:
            h = 5
        if h <= 4:
            return [3, 4], [4, 5, 6, 7, 8], [7, 8, 9]
        if h <= 6:
            return [4, 5], [5, 6, 7, 8], [8, 9]
        return [5, 6], [6, 7, 8], [8, 9, 10]

    plants_js = []
    for i, row in df.iterrows():
        name = str(row.get('common_name', row.get('name', f'Plant {i}')))
        latin = str(row.get('latin_name', row.get('scientific_name', '')))
        score = float(row.get('suitability_score', row.get('score', 0)))
        cluster = int(row.get('cluster', 0))
        shade = str(row.get('shade', 'F'))
        moisture = str(row.get('moisture', 'M'))
        soil = str(row.get('soil', 'M'))
        hardiness = str(row.get('hardiness', '5'))
        habit = str(row.get('habit', ''))
        edibility = str(row.get('edibility', ''))
        emoji = pick_emoji(name, habit, edibility)
        sow, grow, harvest = estimate_sow(hardiness, shade)
        moisture_labels = {'D': 'Dry — well-drained', 'M': 'Moist soil', 'We': 'Wet ok', 'Wa': 'Aquatic'}
        shade_labels = {'F': 'Full sun', 'S': 'Semi-shade', 'N': 'Full shade'}
        plants_js.append({
            'id': f'p{i}',
            'name': name,
            'latin': latin,
            'emoji': emoji,
            'score': round(score, 3),
            'cluster': cluster,
            'shade': shade,
            'moisture': moisture,
            'soil': soil,
            'hardiness': hardiness,
            'sow': sow,
            'grow': grow,
            'harvest': harvest,
            'water': moisture_labels.get(moisture, moisture),
            'sun': shade_labels.get(shade, shade),
            'tip': f'Suitability score: {score:.2f}. Cluster {cluster}.',
            'tags': [habit or 'plant'],
            'source': 'streamlit',
        })

    plants_json = json.dumps(plants_js, ensure_ascii=False)
    impact_json = json.dumps(impact_level)
    garden_json = json.dumps(garden_name)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Garden Planner — {garden_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--soil:#8B6340;--soil-l:#C49A6C;--moss:#4A7C59;--moss-l:#7AAE8A;--moss-p:#D6EBDC;--parch:#FAF7F0;--bark:#5C3D1E;--sun:#E8A020;--cream:#F5EFE0;--tx:#2C1F0E;--txm:#7A5C3A;}}
body{{font-family:system-ui,-apple-system,sans-serif;background:var(--parch);color:var(--tx);}}
.app{{max-width:980px;margin:0 auto;padding:16px}}
.hdr{{text-align:center;padding:18px 0 14px;border-bottom:1px solid #D4C4A0;margin-bottom:14px}}
.hdr h1{{font-family:'Lora',serif;font-size:26px;font-weight:500;color:var(--bark)}}
.hdr p{{font-size:12px;color:var(--txm);font-style:italic;font-family:'Lora',serif;margin-top:3px}}
.tabs{{display:flex;gap:3px;margin-bottom:14px;background:var(--cream);border-radius:10px;padding:4px}}
.tab{{flex:1;padding:8px 4px;border:none;background:transparent;border-radius:7px;font-family:'DM Sans',sans-serif;font-size:12px;cursor:pointer;color:var(--txm);transition:all .15s}}
.tab.active{{background:white;color:var(--bark);font-weight:500;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.sec{{display:none}}.sec.active{{display:block}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}}
.stat{{background:white;border-radius:9px;border:1px solid #D4C4A0;padding:10px;text-align:center}}
.stat-n{{font-size:22px;font-weight:500;color:var(--bark)}}
.stat-l{{font-size:11px;color:var(--txm);margin-top:2px}}
.impact{{padding:8px 12px;border-radius:8px;font-size:12px;margin-bottom:12px;border-left:3px solid}}
.impact-low{{background:#D6EBDC;color:#27500A;border-color:#4A7C59}}
.impact-moderate{{background:#FFF3CC;color:#633806;border-color:#E8A020}}
.impact-high{{background:#FFE0CC;color:#712B13;border-color:#C07020}}
.impact-severe{{background:#FFD0D0;color:#791F1F;border-color:#C0392B}}
.cluster-legend{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}}
.cl-badge{{font-size:11px;padding:3px 9px;border-radius:10px;font-weight:500}}
.planner-layout{{display:grid;grid-template-columns:1fr 230px;gap:14px}}
.panel{{background:white;border-radius:12px;border:1px solid #D4C4A0;padding:12px}}
.panel h3{{font-family:'Lora',serif;font-size:14px;font-weight:500;color:var(--bark);margin-bottom:8px}}
.ctrl{{display:flex;gap:6px;margin-bottom:8px;align-items:center;flex-wrap:wrap;font-size:11px;color:var(--txm)}}
.ctrl select{{font-size:11px;padding:3px 7px;border:1px solid #D4C4A0;border-radius:6px;background:var(--parch);color:var(--tx)}}
.btn-sm{{padding:4px 9px;background:transparent;border:1px solid #D4C4A0;border-radius:6px;font-size:11px;cursor:pointer;color:var(--txm);font-family:'DM Sans',sans-serif}}
.btn-sm:hover{{border-color:var(--soil);color:var(--soil)}}
.garden-grid{{display:grid;gap:2px}}
.cell{{width:100%;aspect-ratio:1;border-radius:4px;border:1px solid #D4C4A0;background:#F0E8D0;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:13px;transition:all .12s;position:relative;overflow:hidden}}
.cell:hover{{border-color:var(--moss);background:var(--moss-p);transform:scale(1.07);z-index:2}}
.cell.occ{{background:var(--moss-p);border-color:var(--moss-l)}}
.cell .cc{{position:absolute;bottom:0;left:0;right:0;height:3px}}
.cell .cn{{position:absolute;top:1px;right:2px;font-size:7px;color:var(--txm)}}
.pal-search{{width:100%;padding:5px 8px;border:1px solid #D4C4A0;border-radius:7px;font-size:12px;background:var(--parch);font-family:'DM Sans',sans-serif;color:var(--tx);margin-bottom:6px}}
.pal-search:focus{{outline:none;border-color:var(--moss)}}
.plant-list{{display:flex;flex-direction:column;gap:3px;max-height:300px;overflow-y:auto;padding-right:3px}}
.pi{{display:flex;align-items:center;gap:7px;padding:6px 8px;border-radius:8px;cursor:pointer;border:1px solid transparent;transition:all .12s}}
.pi:hover{{background:var(--moss-p);border-color:var(--moss-l)}}
.pi.sel{{background:var(--moss-p);border-color:var(--moss)}}
.pe{{font-size:16px;width:22px;text-align:center}}
.pinfo{{flex:1;min-width:0}}
.pn{{font-size:12px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ps{{font-size:10px;color:var(--txm);display:flex;align-items:center;gap:4px;margin-top:2px}}
.sbar{{width:36px;height:5px;border-radius:3px;background:#E0D8C8;overflow:hidden;flex-shrink:0}}
.sfill{{height:100%;border-radius:3px}}
.info-box{{background:var(--cream);border-radius:10px;border:1px solid #D4C4A0;padding:11px;margin-top:10px}}
.info-box h4{{font-family:'Lora',serif;font-size:13px;font-weight:500;color:var(--bark);margin-bottom:6px}}
.tr{{display:flex;gap:5px;align-items:flex-start;margin-bottom:4px;font-size:11px;color:var(--txm)}}
.db-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}}
.pc{{background:white;border-radius:12px;border:1px solid #D4C4A0;padding:12px;cursor:pointer;transition:all .18s}}
.pc:hover{{border-color:var(--moss);transform:translateY(-1px)}}
.pce{{font-size:28px;margin-bottom:5px}}
.pcn{{font-family:'Lora',serif;font-size:13px;font-weight:500;color:var(--bark)}}
.pcl{{font-size:10px;color:var(--txm);font-style:italic;margin-bottom:6px}}
.pcsc{{display:flex;align-items:center;gap:5px;margin-bottom:5px}}
.pcsb{{flex:1;height:4px;border-radius:2px;background:#E0D8C8;overflow:hidden}}
.pcsf{{height:100%;border-radius:2px}}
.tags{{display:flex;flex-wrap:wrap;gap:3px}}
.tag{{font-size:10px;padding:2px 6px;border-radius:4px;border:1px solid #D4C4A0;background:var(--cream);color:var(--txm)}}
.ctag{{font-size:10px;padding:2px 7px;border-radius:10px;font-weight:500}}
.db-detail{{background:white;border-radius:12px;border:1px solid #D4C4A0;padding:14px}}
.back{{background:transparent;border:1px solid #D4C4A0;border-radius:7px;padding:4px 9px;font-size:11px;cursor:pointer;color:var(--txm);font-family:'DM Sans',sans-serif;margin-bottom:10px}}
.back:hover{{color:var(--bark);border-color:var(--soil)}}
.care-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.ci{{background:var(--cream);border-radius:7px;padding:8px 10px}}
.ci-l{{font-size:10px;color:var(--txm);margin-bottom:2px}}
.ci-v{{font-size:12px;font-weight:500;color:var(--tx)}}
.cal-wrap{{overflow-x:auto}}
.cal-grid{{display:grid;gap:2px;min-width:580px}}
.cal-hdr{{padding:4px 3px;text-align:center;font-weight:500;color:var(--txm);font-size:10px}}
.cal-nm{{padding:5px 7px;font-size:11px;display:flex;align-items:center;gap:5px;white-space:nowrap}}
.cal-cell{{border-radius:2px;height:22px}}
.csow{{background:#C8E6C9}}.cgrow{{background:#7AAE8A}}.charv{{background:#E8A020;opacity:.8}}.cnone{{background:var(--cream)}}
.cal-legend{{display:flex;gap:14px;margin-bottom:10px;flex-wrap:wrap}}
.cl-item{{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--txm)}}
.cl-dot{{width:12px;height:12px;border-radius:2px;flex-shrink:0}}
.add-form{{background:white;border-radius:12px;border:1px solid #D4C4A0;padding:12px;margin-bottom:12px}}
.form-row{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}}
.fg{{display:flex;flex-direction:column;gap:3px;flex:1;min-width:110px}}
.fg label{{font-size:11px;color:var(--txm)}}
.fg select,.fg input{{padding:6px 8px;border:1px solid #D4C4A0;border-radius:7px;font-size:12px;background:var(--parch);color:var(--tx);font-family:'DM Sans',sans-serif}}
.save-btn{{padding:7px 16px;background:var(--moss);border:none;border-radius:8px;color:white;font-size:12px;cursor:pointer;font-family:'DM Sans',sans-serif;font-weight:500}}
.save-btn:hover{{background:var(--bark)}}
.recs{{display:flex;flex-direction:column;gap:6px}}
.rec{{background:white;border-radius:9px;border:1px solid #D4C4A0;padding:10px 12px;display:flex;align-items:center;gap:10px}}
.empty{{text-align:center;padding:30px;color:var(--txm);font-style:italic;font-family:'Lora',serif;font-size:13px}}
::-webkit-scrollbar{{width:4px}}::-webkit-scrollbar-thumb{{background:#C4B090;border-radius:2px}}
</style>
</head>
<body>
<div class="app">
<div class="hdr">
  <h1>🌱 {garden_name}</h1>
  <p id="sub">Visual Garden Planner — generated by your Streamlit analysis</p>
</div>
<div class="tabs">
  <button class="tab active" onclick="sw('overview')">Overview</button>
  <button class="tab" onclick="sw('planner')">Garden Grid</button>
  <button class="tab" onclick="sw('library')">Plant Library</button>
  <button class="tab" onclick="sw('calendar')">Calendar</button>
  <button class="tab" onclick="sw('myplants')">My Garden</button>
</div>

<!-- OVERVIEW -->
<div id="tab-overview" class="sec active">
  <div class="stats">
    <div class="stat"><div class="stat-n" id="ov-plants">—</div><div class="stat-l">Plants</div></div>
    <div class="stat"><div class="stat-n" id="ov-clusters">—</div><div class="stat-l">Clusters</div></div>
    <div class="stat"><div class="stat-n" id="ov-top">—</div><div class="stat-l">Top score</div></div>
    <div class="stat"><div class="stat-n" id="ov-avg">—</div><div class="stat-l">Avg score</div></div>
  </div>
  <div id="ov-impact" style="display:none"></div>
  <div id="ov-legend" class="cluster-legend"></div>
  <p style="font-size:12px;color:var(--txm);line-height:1.6">
    This planner was pre-loaded with your Streamlit recommendation results.
    Use the <strong>Garden Grid</strong> tab to drag plants onto your plot,
    <strong>Plant Library</strong> to explore all recommendations,
    <strong>Calendar</strong> for seasonal timing, and
    <strong>My Garden</strong> to track what you've actually sown.
  </p>
</div>

<!-- GRID -->
<div id="tab-planner" class="sec">
  <div class="planner-layout">
    <div class="panel">
      <h3>Garden plot</h3>
      <div class="ctrl">
        <span>Size:</span>
        <select id="gSize" onchange="resizeGrid()">
          <option value="6">6×6 small</option>
          <option value="8" selected>8×8 medium</option>
          <option value="10">10×10 large</option>
        </select>
        <span style="margin-left:4px">Cluster:</span>
        <select id="clFilter" onchange="buildPal()"><option value="all">All</option></select>
        <button class="btn-sm" style="margin-left:auto" onclick="clearGrid()">Clear</button>
      </div>
      <div id="gGrid" class="garden-grid"></div>
    </div>
    <div>
      <div class="panel">
        <h3>Choose a plant</h3>
        <input class="pal-search" type="text" placeholder="Search…" oninput="buildPal(this.value)" id="palQ">
        <div class="plant-list" id="palList"></div>
      </div>
      <div id="infoBox" style="display:none" class="info-box"></div>
    </div>
  </div>
</div>

<!-- LIBRARY -->
<div id="tab-library" class="sec">
  <div style="display:flex;gap:8px;margin-bottom:12px">
    <input type="text" placeholder="Search plants…" oninput="filterLib(this.value)"
      style="flex:1;padding:7px 10px;border:1px solid #D4C4A0;border-radius:8px;font-size:13px;background:white;color:var(--tx)">
    <select onchange="sortLib(this.value)"
      style="padding:7px 10px;border:1px solid #D4C4A0;border-radius:8px;font-size:12px;background:var(--parch);color:var(--tx)">
      <option value="score">By score</option>
      <option value="name">By name</option>
      <option value="cluster">By cluster</option>
    </select>
  </div>
  <div id="libList"></div>
  <div id="libDetail" style="display:none"></div>
</div>

<!-- CALENDAR -->
<div id="tab-calendar" class="sec">
  <div class="cal-legend">
    <div class="cl-item"><div class="cl-dot" style="background:#C8E6C9"></div>Sow</div>
    <div class="cl-item"><div class="cl-dot" style="background:#7AAE8A"></div>Grow</div>
    <div class="cl-item"><div class="cl-dot" style="background:#E8A020;opacity:.8"></div>Harvest</div>
  </div>
  <div class="cal-wrap"><div id="calGrid" class="cal-grid"></div></div>
</div>

<!-- MY GARDEN -->
<div id="tab-myplants" class="sec">
  <div class="add-form">
    <div class="form-row">
      <div class="fg"><label>Plant</label><select id="rPlant"></select></div>
      <div class="fg"><label>Bed / location</label><input type="text" id="rLoc" placeholder="e.g. Raised Bed A"></div>
      <div class="fg"><label>Date planted</label><input type="date" id="rDate"></div>
      <div class="fg"><label>Status</label>
        <select id="rStatus">
          <option value="planted">Just planted</option>
          <option value="growing">Growing</option>
          <option value="harvested">Harvested</option>
        </select>
      </div>
    </div>
    <button class="save-btn" onclick="addRec()">+ Add to my garden</button>
  </div>
  <div class="recs" id="recList"></div>
</div>
</div>

<script id="plant-data" type="application/json">{plants_json}</script>
<script id="meta-data" type="application/json">{{"impact":{impact_json},"garden":{garden_json}}}</script>
<script>
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const CC=['#4A7C59','#7B5EA7','#C07020','#2271B3','#C0392B','#1A7A6A','#8B6340','#5C7A2A','#B03060','#2C5F8A'];
const PLANTS=JSON.parse(document.getElementById('plant-data').textContent);
const _meta=JSON.parse(document.getElementById('meta-data').textContent);
const IMPACT=_meta.impact;
const GARDEN=_meta.garden;

let sel=null,gData={{}},gSize=8,recs=[],libSort='score';
const cc=i=>CC[i%CC.length];
const sc=s=>s>=.8?'#3B6D11':s>=.6?'#BA7517':'#993C1D';
const sl=s=>s>=.8?'Excellent':s>=.6?'Good':'Fair';

function sw(name){{
  const ns=['overview','planner','library','calendar','myplants'];
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',ns[i]===name));
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
}}

function init(){{
  const scores=PLANTS.map(p=>p.score).filter(s=>s>0);
  const clusters=[...new Set(PLANTS.map(p=>p.cluster))].sort((a,b)=>a-b);
  document.getElementById('ov-plants').textContent=PLANTS.length;
  document.getElementById('ov-clusters').textContent=clusters.length;
  document.getElementById('ov-top').textContent=scores.length?Math.max(...scores).toFixed(2):'—';
  document.getElementById('ov-avg').textContent=scores.length?(scores.reduce((a,b)=>a+b)/scores.length).toFixed(2):'—';
  if(IMPACT){{
    const d=document.getElementById('ov-impact');
    d.style.display='block';
    d.className='impact impact-'+IMPACT;
    d.textContent='Climate impact level for this location: '+IMPACT.charAt(0).toUpperCase()+IMPACT.slice(1)+' (IPCC regional projection, 5 years)';
  }}
  const leg=document.getElementById('ov-legend');
  clusters.forEach(c=>{{
    const cnt=PLANTS.filter(p=>p.cluster===c).length;
    const sp=document.createElement('span');
    sp.className='cl-badge';
    sp.style=`background:${{cc(c)}}22;color:${{cc(c)}};border:1px solid ${{cc(c)}}55`;
    sp.textContent='Cluster '+c+' ('+cnt+' plants)';
    leg.appendChild(sp);
  }});
  buildGrid();buildPal();buildLib();buildCal();
  buildClusterDropdown();buildRecForm();
  document.getElementById('rDate').value=new Date().toISOString().split('T')[0];
}}

function buildClusterDropdown(){{
  const sel=document.getElementById('clFilter');
  const cls=[...new Set(PLANTS.map(p=>p.cluster))].sort((a,b)=>a-b);
  cls.forEach(c=>{{const o=document.createElement('option');o.value=c;o.textContent='Cluster '+c;sel.appendChild(o);}});
}}

function buildGrid(){{
  const g=document.getElementById('gGrid');
  g.style.gridTemplateColumns='repeat('+gSize+',1fr)';
  g.innerHTML='';
  for(let r=0;r<gSize;r++) for(let c=0;c<gSize;c++){{
    const key=r+','+c,div=document.createElement('div');
    div.className='cell'+(gData[key]?' occ':'');
    div.onclick=()=>cellClick(key,div);
    if(gData[key]){{
      const p=PLANTS.find(x=>x.id===gData[key]);
      if(p) div.innerHTML='<span>'+p.emoji+'</span><span class="cn">C'+p.cluster+'</span><div class="cc" style="background:'+cc(p.cluster)+'"></div>';
    }}
    g.appendChild(div);
  }}
}}
function cellClick(key,div){{
  if(!sel){{ if(gData[key]){{ delete gData[key];div.className='cell';div.innerHTML=''; }} return; }}
  gData[key]=sel.id;
  div.className='cell occ';
  div.innerHTML='<span>'+sel.emoji+'</span><span class="cn">C'+sel.cluster+'</span><div class="cc" style="background:'+cc(sel.cluster)+'"></div>';
}}
function resizeGrid(){{ gSize=parseInt(document.getElementById('gSize').value);buildGrid(); }}
function clearGrid(){{ gData={{}};buildGrid(); }}

function buildPal(q){{
  const f=(q||document.getElementById('palQ').value||'').toLowerCase();
  const cf=document.getElementById('clFilter').value;
  const list=document.getElementById('palList');
  list.innerHTML='';
  PLANTS.filter(p=>(!f||p.name.toLowerCase().includes(f))&&(cf==='all'||p.cluster===parseInt(cf)))
    .sort((a,b)=>b.score-a.score).forEach(p=>{{
      const d=document.createElement('div');
      d.className='pi'+(sel&&sel.id===p.id?' sel':'');
      const sw=Math.round(p.score*36);
      d.innerHTML='<span class="pe">'+p.emoji+'</span><div class="pinfo">'+
        '<div class="pn">'+p.name+'</div>'+
        '<div class="ps"><span style="font-size:9px;padding:1px 5px;border-radius:8px;background:'+cc(p.cluster)+'22;color:'+cc(p.cluster)+';border:1px solid '+cc(p.cluster)+'44">C'+p.cluster+'</span>'+
        '<span style="font-size:10px;color:'+sc(p.score)+'">'+sl(p.score)+'</span></div></div>'+
        '<div class="sbar"><div class="sfill" style="width:'+sw+'px;background:'+sc(p.score)+'"></div></div>';
      d.onclick=()=>selectPlant(p);
      list.appendChild(d);
    }});
}}
function selectPlant(p){{
  sel=sel&&sel.id===p.id?null:p;
  buildPal();
  const box=document.getElementById('infoBox');
  if(sel){{
    box.style.display='block';
    box.innerHTML='<h4>'+p.emoji+' '+p.name+'</h4>'+
      '<div class="tr"><span>💧</span><span>'+p.water+'</span></div>'+
      '<div class="tr"><span>☀️</span><span>'+p.sun+'</span></div>'+
      '<div class="tr"><span>🤝</span><span style="color:'+cc(p.cluster)+'">Cluster '+p.cluster+' companion group</span></div>'+
      '<div class="tr"><span>📊</span><span style="color:'+sc(p.score)+'">Score: '+p.score.toFixed(2)+' — '+sl(p.score)+'</span></div>';
  }} else box.style.display='none';
}}

function buildLib(q,sort){{
  if(sort) libSort=sort;
  const f=(q||'').toLowerCase();
  const lv=document.getElementById('libList'),dv=document.getElementById('libDetail');
  lv.style.display='block';dv.style.display='none';
  let filtered=PLANTS.filter(p=>!f||p.name.toLowerCase().includes(f)||p.latin.toLowerCase().includes(f));
  if(libSort==='score') filtered.sort((a,b)=>b.score-a.score);
  else if(libSort==='name') filtered.sort((a,b)=>a.name.localeCompare(b.name));
  else filtered.sort((a,b)=>a.cluster-b.cluster||b.score-a.score);
  lv.innerHTML='<div class="db-grid">'+filtered.map(p=>
    '<div class="pc" onclick="showDetail(\''+p.id+'\')">'+
    '<div class="pce">'+p.emoji+'</div>'+
    '<div class="pcn">'+p.name+'</div>'+
    '<div class="pcl">'+p.latin+'</div>'+
    '<div class="pcsc"><span style="font-size:10px;color:var(--txm)">'+p.score.toFixed(2)+'</span>'+
    '<div class="pcsb"><div class="pcsf" style="width:'+Math.round(p.score*100)+'%;background:'+sc(p.score)+'"></div></div></div>'+
    '<div class="tags"><span class="ctag" style="background:'+cc(p.cluster)+'22;color:'+cc(p.cluster)+';border:1px solid '+cc(p.cluster)+'44">C'+p.cluster+'</span>'+
    (p.tags||[]).slice(0,1).map(t=>'<span class="tag">'+t+'</span>').join('')+'</div></div>'
  ).join('')+'</div>';
}}
function filterLib(v){{ buildLib(v); }}
function sortLib(v){{ buildLib(null,v); }}
function showDetail(id){{
  const p=PLANTS.find(x=>x.id===id);if(!p)return;
  document.getElementById('libList').style.display='none';
  const dv=document.getElementById('libDetail');
  dv.style.display='block';
  dv.innerHTML='<button class="back" onclick="buildLib()">← Back</button><div class="db-detail">'+
    '<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #D4C4A0">'+
    '<div style="font-size:38px">'+p.emoji+'</div>'+
    '<div><div style="font-family:Lora,serif;font-size:20px;font-weight:500;color:var(--bark)">'+p.name+'</div>'+
    '<div style="font-style:italic;color:var(--txm);font-size:12px">'+p.latin+'</div></div></div>'+
    '<div class="care-grid">'+
    '<div class="ci"><div class="ci-l">💧 Watering</div><div class="ci-v">'+p.water+'</div></div>'+
    '<div class="ci"><div class="ci-l">☀️ Sunlight</div><div class="ci-v">'+p.sun+'</div></div>'+
    '<div class="ci"><div class="ci-l">🤝 Cluster</div><div class="ci-v" style="color:'+cc(p.cluster)+'">Group '+p.cluster+'</div></div>'+
    '<div class="ci"><div class="ci-l">📊 Score</div><div class="ci-v" style="color:'+sc(p.score)+'">'+p.score.toFixed(2)+' — '+sl(p.score)+'</div></div>'+
    '<div class="ci"><div class="ci-l">🌱 Sow</div><div class="ci-v">'+p.sow.map(m=>MONTHS[m-1]).join(', ')+'</div></div>'+
    '<div class="ci"><div class="ci-l">🌾 Harvest</div><div class="ci-v">'+p.harvest.map(m=>MONTHS[m-1]).join(', ')+'</div></div>'+
    '</div></div>';
}}

function buildCal(){{
  const cg=document.getElementById('calGrid');
  cg.style.gridTemplateColumns='160px repeat(12,1fr)';
  cg.innerHTML='<div class="cal-hdr"></div>'+MONTHS.map(m=>'<div class="cal-hdr">'+m+'</div>').join('');
  PLANTS.slice(0,20).forEach(p=>{{
    const nd=document.createElement('div');
    nd.className='cal-nm';
    nd.innerHTML='<span style="width:8px;height:8px;border-radius:50%;background:'+cc(p.cluster)+';flex-shrink:0;display:inline-block"></span>'+
      '<span>'+p.emoji+'</span><span style="overflow:hidden;text-overflow:ellipsis">'+p.name+'</span>';
    cg.appendChild(nd);
    for(let m=1;m<=12;m++){{
      const cell=document.createElement('div');
      cell.className='cal-cell '+(p.harvest.includes(m)?'charv':p.grow.includes(m)?'cgrow':p.sow.includes(m)?'csow':'cnone');
      cg.appendChild(cell);
    }}
  }});
}}

function buildRecForm(){{
  const sel=document.getElementById('rPlant');
  sel.innerHTML=PLANTS.map(p=>'<option value="'+p.id+'">'+p.emoji+' '+p.name+'</option>').join('');
  renderRecs();
}}
function addRec(){{
  const pid=document.getElementById('rPlant').value,p=PLANTS.find(x=>x.id===pid);
  recs.push({{id:Date.now(),pid,loc:document.getElementById('rLoc').value||'Garden',
    date:document.getElementById('rDate').value,status:document.getElementById('rStatus').value}});
  renderRecs();
}}
function delRec(id){{ recs=recs.filter(r=>r.id!==id);renderRecs(); }}
function renderRecs(){{
  const list=document.getElementById('recList');
  if(!recs.length){{ list.innerHTML='<div class="empty">No plants added yet.</div>';return; }}
  const sc2={{planted:'#E8A020',growing:'#4A7C59',harvested:'#C49A6C'}};
  list.innerHTML=recs.map(r=>{{
    const p=PLANTS.find(x=>x.id===r.pid)||{{emoji:'🌱',name:r.pid,cluster:0}};
    return '<div class="rec"><div style="font-size:20px;width:28px;text-align:center">'+p.emoji+'</div>'+
      '<div style="flex:1"><div style="font-size:13px;font-weight:500">'+p.name+
      ' <span class="ctag" style="font-size:10px;background:'+cc(p.cluster)+'22;color:'+cc(p.cluster)+';border:1px solid '+cc(p.cluster)+'44">C'+p.cluster+'</span></div>'+
      '<div style="font-size:10px;color:var(--txm)">'+r.date+' · '+r.status+'</div></div>'+
      '<span style="font-size:11px;background:var(--moss-p);color:var(--moss);border-radius:4px;padding:2px 6px;border:1px solid var(--moss-l)">'+r.loc+'</span>'+
      '<div style="width:7px;height:7px;border-radius:50%;background:'+(sc2[r.status]||'#888')+'"></div>'+
      '<button onclick="delRec('+r.id+')" style="background:none;border:none;cursor:pointer;color:#C49A6C;font-size:16px;padding:0 4px">×</button></div>';
  }}).join('');
}}

document.addEventListener('DOMContentLoaded', function() {{
  try {{ init(); }} catch(e) {{ document.body.innerHTML='<pre style="color:red;padding:20px">Error loading planner: '+e.message+'</pre>'; }}
}});
</script>
</body>
</html>"""
    return html


# ── Session state ─────────────────────────────────────────────────────────────

if 'results' not in st.session_state:
    st.session_state.results = None
if 'climate_projection' not in st.session_state:
    st.session_state.climate_projection = None

# ── Generate ──────────────────────────────────────────────────────────────────

if generate:
    st.session_state.results = None
    st.session_state.climate_projection = None

    plant_db = "pfaf2.csv"
    companion_db = "companion_plants.csv"

    if not Path(plant_db).exists():
        st.error(f"❌ Error: Plant database '{plant_db}' not found!")
        st.stop()

    companion_available = Path(companion_db).exists()
    if not companion_available:
        st.warning(f"⚠️ Companion plants database '{companion_db}' not found. Companion analysis will be skipped.")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🧹 Cleaning old files...")
        progress_bar.progress(5)
        for f in os.listdir('.'):
            if '_recommendations.csv' in f or '_results.xlsx' in f or 'plant_clusters' in f:
                try: os.remove(f)
                except: pass

        Config.MAX_CLUSTER_SIZE = max_cluster

        status_text.text("🚀 Initializing Garden Planner...")
        progress_bar.progress(10)
        planner = GardenPlanner(use_vectorized=True)
        planner.initialize(plant_db)

        status_text.text("📍 Fetching location data...")
        progress_bar.progress(30)
        location_id = planner.add_location(latitude, longitude, garden_name)

        status_text.text("🌱 Calculating plant recommendations...")
        progress_bar.progress(50)
        recommendations = planner.get_recommendations(location_id, num_rec, min_score)

        if recommendations.empty:
            progress_bar.empty(); status_text.empty()
            st.warning("⚠️ No suitable plants found. Try lowering the minimum suitability score.")
            st.stop()

        status_text.text("💾 Saving recommendations...")
        progress_bar.progress(60)
        csv_filename = f"{garden_name.replace(' ', '_')}_recommendations.csv"
        recommendations.to_csv(csv_filename, index=False)

        status_text.text("🔬 Clustering plants...")
        progress_bar.progress(70)
        clustered_df = PlantClusteringModule.cluster_plants(recommendations, max_cluster)

        status_text.text("📊 Creating visualizations...")
        progress_bar.progress(75)
        fig = PlantClusteringModule.visualize_clusters(clustered_df, garden_name)

        cluster_companions = {}
        if companion_available:
            status_text.text("🤝 Analyzing companion relationships...")
            progress_bar.progress(80)
            cluster_companions = PlantClusteringModule.find_companions(clustered_df, companion_db)

        status_text.text("📊 Generating Excel report...")
        progress_bar.progress(85)
        excel_filename = f"{garden_name.replace(' ', '_')}_results.xlsx"
        PlantClusteringModule.export_to_excel(clustered_df, cluster_companions, fig, garden_name, excel_filename)

        status_text.text("🌍 Generating climate projections...")
        progress_bar.progress(90)
        climate_projection_data = None
        with planner.db.get_connection() as conn:
            climate_data = pd.read_sql(
                f"SELECT * FROM climate_data WHERE location_id = {location_id} AND scenario = 'current' LIMIT 1",
                conn
            )
            if not climate_data.empty:
                climate_row = climate_data.iloc[0]
                projection, summary = get_climate_projection_for_location(
                    latitude, longitude,
                    float(climate_row['avg_temp']),
                    float(climate_row['precip']),
                    int(climate_row['frost_days']),
                    garden_name
                )
                climate_projection_data = {'projection': projection, 'summary': summary}
                st.session_state.climate_projection = climate_projection_data

        status_text.text("🔍 Collecting results...")
        progress_bar.progress(95)
        png_files = [f for f in os.listdir('.') if 'plant_cluster' in f and f.endswith('.png')]

        # Build standalone HTML planner
        impact_level = ""
        if climate_projection_data:
            impact_level = climate_projection_data['projection'].impact_level
        planner_html = build_planner_html(clustered_df, garden_name, impact_level)
        html_filename = f"{garden_name.replace(' ', '_')}_garden_planner.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(planner_html)

        st.session_state.results = {
            'df': clustered_df,
            'csv': csv_filename,
            'xlsx': excel_filename,
            'html': html_filename,
            'png': png_files,
            'garden_name': garden_name,
            'location': f"{latitude}, {longitude}",
            'num_clusters': clustered_df['cluster'].nunique(),
            'num_companions': sum(len(d) for d in cluster_companions.values()) if cluster_companions else 0,
        }

        status_text.text("✅ Complete!")
        progress_bar.progress(100)
        import time; time.sleep(0.5)
        status_text.empty(); progress_bar.empty()

        st.success("✅ Garden plan generated successfully!")
        st.rerun()

    except Exception as e:
        progress_bar.empty(); status_text.empty()
        st.error(f"❌ Error generating garden plan: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

# ── Display results ───────────────────────────────────────────────────────────

if st.session_state.results:
    df = st.session_state.results['df']

    # Climate Projection
    if st.session_state.climate_projection:
        st.markdown("---")
        st.markdown("### 🌍 Climate Change Projection (5 Years)")

        proj = st.session_state.climate_projection['projection']
        summary = st.session_state.climate_projection['summary']

        impact_colors = {'low': '#4CAF50', 'moderate': '#FF9800', 'high': '#F44336', 'severe': '#D32F2F'}
        impact_color = impact_colors.get(proj.impact_level, '#FF9800')

        st.markdown(f"""
        <div style='background-color: {impact_color}20; border-left: 4px solid {impact_color}; padding: 1rem; border-radius: 5px;'>
            <h4 style='color: {impact_color}; margin: 0;'>
                Impact Level: {proj.impact_level.upper()}
                <span style='font-size: 0.8em; color: #666;'>(Confidence: {proj.confidence})</span>
            </h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Temperature Change", f"+{proj.temp_change}°C", f"{proj.temp_change_min} to {proj.temp_change_max}°C", delta_color="inverse")
        with col2:
            st.metric("Precipitation Change", f"{proj.precip_change:+.1f}%", f"{proj.precip_change_min:+.1f} to {proj.precip_change_max:+.1f}%")
        with col3:
            st.metric("Growing Season", f"{proj.growing_season_change:+d} days", "Longer season" if proj.growing_season_change > 0 else "Shorter")
        with col4:
            st.metric("Hardiness Zone Shift", f"+{proj.hardiness_zone_shift:.1f} zones", "Warmer zones")

        with st.expander("📊 Detailed Climate Impacts", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🌡️ Temperature:**"); st.info(summary['temperature'])
                st.markdown("**💧 Precipitation:**"); st.info(summary['precipitation'])
                st.markdown("**📅 Growing Season:**"); st.info(summary['growing_season'])
            with col2:
                st.markdown("**⚠️ Extreme Events:**"); st.warning(summary['extreme_events'])
                st.markdown("**🌱 Gardening Implications:**"); st.success(summary['gardening_implications'])

        st.markdown("#### 🌿 What This Means for Your Garden")
        recs_text = []
        if proj.temp_change > 1.0:
            recs_text.append("🌡️ **Consider heat-tolerant varieties** — your location will be warmer")
        if proj.precip_change < -3:
            recs_text.append("💧 **Focus on drought-resistant plants** — reduced rainfall expected")
        elif proj.precip_change > 3:
            recs_text.append("💧 **Ensure good drainage** — increased rainfall expected")
        if proj.growing_season_change > 10:
            recs_text.append("📅 **Extended growing season** — longer-season crops become viable")
        if proj.hardiness_zone_shift >= 0.5:
            recs_text.append(f"🗺️ **Hardiness zone shift** — effectively {proj.hardiness_zone_shift:.1f} zones warmer")
        if proj.heat_wave_increase > 20:
            recs_text.append("☀️ **Prepare for more heat waves** — consider shade structures and mulching")
        for r in recs_text:
            st.markdown(r)
        if not recs_text:
            st.info("✅ Your climate is expected to remain relatively stable for gardening purposes")

    st.markdown("---")

    # Summary metrics
    st.markdown("### 📊 Garden Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Plants", len(df))
    with col2:
        score_col = next((c for c in ['suitability_score', 'score'] if c in df.columns), None)
        if score_col:
            st.metric("Avg Suitability", f"{df[score_col].mean():.2f}")
    with col3:
        st.metric("Clusters", st.session_state.results['num_clusters'])
    with col4:
        st.metric("Location", st.session_state.results['location'])

    st.markdown("---")

    # ── TABS for results ─────────────────────────────────────────────────────
    tab_recs, tab_planner, tab_viz, tab_fields = st.tabs([
        "🌿 Recommendations",
        "🗺️ Visual Garden Planner",
        "📊 Cluster Visualization",
        "📖 Field Guide",
    ])

    # ── Tab 1: Recommendations ───────────────────────────────────────────────
    with tab_recs:
        st.markdown("### 🌿 Top Plant Recommendations")
        score_col = next((c for c in ['suitability_score', 'Suitability Score', 'score', 'Score'] if c in df.columns), None)

        if score_col:
            top_plants = df.nlargest(10, score_col)
            for idx, (_, row) in enumerate(top_plants.iterrows(), 1):
                with st.container():
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        common_name = row.get('common_name', row.get('Common Name', row.get('name', 'Unknown')))
                        latin_name = row.get('latin_name', row.get('Latin Name', row.get('scientific_name', '')))
                        cluster_id = int(row.get('cluster', 0))
                        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

                        st.markdown(f"**{idx}. {common_name}**")
                        if latin_name:
                            st.caption(f"*{latin_name}*")

                        # Cluster badge + companion info
                        companion_plants = []
                        if 'cluster' in df.columns:
                            cluster_mates = df[df['cluster'] == cluster_id]
                            companion_plants = [
                                str(r.get('common_name', r.get('name', '')))
                                for _, r in cluster_mates.iterrows()
                                if str(r.get('common_name', r.get('name', ''))) != str(common_name)
                            ][:4]

                        badge_html = f"<span class='cluster-badge' style='background:{color}22;color:{color};border:1px solid {color}55'>Cluster {cluster_id}</span>"
                        if companion_plants:
                            badge_html += f" <span style='font-size:0.8em;color:#666'>🤝 Grows well with: {', '.join(companion_plants)}</span>"
                        st.markdown(badge_html, unsafe_allow_html=True)

                        detail_cols = st.columns(3)
                        with detail_cols[0]:
                            shade = row.get('shade', row.get('Shade', ''))
                            if shade and shade in FIELD_EXPLANATIONS['Shade']:
                                st.caption(f"☀️ {FIELD_EXPLANATIONS['Shade'][shade]}")
                            else:
                                st.caption("☀️ Shade info not available")
                        with detail_cols[1]:
                            moisture = row.get('moisture', row.get('Moisture', ''))
                            if moisture and moisture in FIELD_EXPLANATIONS['Moisture']:
                                st.caption(f"💧 {FIELD_EXPLANATIONS['Moisture'][moisture]}")
                            else:
                                st.caption("💧 Moisture info not available")
                        with detail_cols[2]:
                            growth = row.get('growth_rate', row.get('Growth Rate', ''))
                            if growth:
                                st.caption(f"📈 Growth: {growth}")
                            else:
                                st.caption("📈 Growth info not available")

                    with col2:
                        score = row[score_col]
                        if score >= 0.8:
                            color_score, label = "#2E7D32", "Excellent"
                        elif score >= 0.6:
                            color_score, label = "#558B2F", "Good"
                        else:
                            color_score, label = "#FFA000", "Fair"
                        st.markdown(
                            f"<div class='metric-container'><div style='font-size:2rem;color:{color_score};font-weight:bold'>{score:.2f}</div>"
                            f"<div style='font-size:0.8rem;color:{color_score}'>{label}</div></div>",
                            unsafe_allow_html=True
                        )
                    st.markdown("---")
        else:
            st.dataframe(df.head(10), use_container_width=True)

    # ── Tab 2: Visual Garden Planner ─────────────────────────────────────────
    with tab_planner:
        st.markdown("### 🗺️ Visual Garden Planner")
        st.info(
            "An interactive garden grid pre-loaded with your recommended plants. "
            "Click a plant in the palette, then click any grid cell to place it. "
            "Colour-coded clusters show companion planting groups at a glance."
        )

        impact_level = ""
        if st.session_state.climate_projection:
            impact_level = st.session_state.climate_projection['projection'].impact_level

        planner_html = build_planner_html(df, garden_name, impact_level)
        components.html(planner_html, height=950, scrolling=True)

        # Download the standalone HTML
        st.markdown("---")
        st.markdown("**💾 Save as standalone file** — open it in any browser, no internet needed:")
        html_path = st.session_state.results.get('html', '')
        if html_path and os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.download_button(
                label="📥 Download Garden Planner (HTML)",
                data=html_content,
                file_name=html_path,
                mime="text/html",
                use_container_width=False
            )
            st.caption("A self-contained HTML file — share it or open it offline anytime.")

    # ── Tab 3: Cluster Visualization ─────────────────────────────────────────
    with tab_viz:
        if st.session_state.results['png']:
            st.markdown("### 📊 Plant Cluster Visualization")
            st.info("""
            **Understanding the Visualization:**
            - 🎨 **Colors** represent different plant clusters
            - 🤝 **Same cluster** = plants grow well together (companion planting)
            - 📏 **Distance** = similarity in growing requirements
            """)
            for png_file in st.session_state.results['png']:
                if os.path.exists(png_file):
                    st.image(add_legend_to_image(png_file), use_column_width=True)
        else:
            st.info("No cluster visualization available for this run.")

    # ── Tab 4: Field Guide ────────────────────────────────────────────────────
    with tab_fields:
        with st.expander("🌤️ Shade Requirements", expanded=True):
            for code, desc in FIELD_EXPLANATIONS['Shade'].items():
                st.markdown(f"- **{code}**: {desc}")
        with st.expander("💧 Moisture Requirements"):
            for code, desc in FIELD_EXPLANATIONS['Moisture'].items():
                st.markdown(f"- **{code}**: {desc}")
        with st.expander("🌱 Soil Type"):
            for code, desc in FIELD_EXPLANATIONS['Soil'].items():
                st.markdown(f"- **{code}**: {desc}")

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.markdown("### 📥 Download Your Garden Plan")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv_path = st.session_state.results.get('csv', '')
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, 'rb') as f:
                st.download_button("📄 Plant List (CSV)", f, file_name=csv_path, mime="text/csv", use_container_width=True)
            st.caption("Spreadsheet format")

    with col2:
        xlsx_path = st.session_state.results.get('xlsx', '')
        if xlsx_path and os.path.exists(xlsx_path):
            with open(xlsx_path, 'rb') as f:
                st.download_button("📊 Full Report (Excel)", f, file_name=xlsx_path,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            st.caption("With clusters & companion info")

    with col3:
        html_path = st.session_state.results.get('html', '')
        if html_path and os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                st.download_button("🗺️ Garden Planner (HTML)", f.read(), file_name=html_path,
                    mime="text/html", use_container_width=True)
            st.caption("Interactive offline planner")

    with col4:
        if st.session_state.results['png']:
            png_file = st.session_state.results['png'][0]
            if os.path.exists(png_file):
                st.download_button("🖼️ Cluster Chart (PNG)", add_legend_to_image(png_file),
                    file_name=f"{garden_name.replace(' ', '_')}_clusters.png", mime="image/png", use_container_width=True)
                st.caption("Cluster diagram")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 View Complete Plant Database"):
        st.dataframe(df, use_container_width=True, height=400)

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome to Garden Planner!

    Get personalised plant recommendations based on your location's real environmental data:

    - 🌡️ **Climate Analysis** — Temperature, rainfall, and hardiness zones
    - 🌍 **Climate Projections** — Expected changes in the next 5 years
    - 🗺️ **Soil Assessment** — pH levels and soil composition
    - 🤝 **Companion Planting** — Plants that grow well together
    - 🗺️ **Visual Grid Planner** — Drag plants onto your garden plot

    #### How to Get Started:
    1. **📍 Enter Your Location** — Use your garden's coordinates
    2. **⚙️ Adjust Preferences** — Set the number of plants and suitability threshold
    3. **🌿 Generate** — Click the button to create your personalised garden plan
    4. **🗺️ Plan Visually** — Use the Visual Garden Planner tab to lay out your plot
    5. **📥 Download** — Save results as CSV, Excel, HTML planner, or PNG
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🎯 Smart Scoring
        - Hardiness match
        - Soil compatibility
        - Climate suitability
        - Water requirements
        """)
    with col2:
        st.markdown("""
        #### 🌍 Climate Projections
        - Temperature trends
        - Rainfall patterns
        - Growing season length
        - Plant zone shifts
        """)
    with col3:
        st.markdown("""
        #### 🗺️ Visual Planner
        - Drag-and-drop garden grid
        - Colour-coded companion clusters
        - Suitability score bars
        - Downloadable offline HTML
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌱 <strong>Garden Planner</strong> • Powered by real environmental data and climate science</p>
    <p style="font-size: 0.85rem;">Data sources: Climate records, PFAF plant database, IPCC projections</p>
</div>
""", unsafe_allow_html=True)
