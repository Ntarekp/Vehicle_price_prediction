import pandas as pd
import json
import folium
from folium import GeoJson, GeoJsonTooltip, GeoJsonPopup
import branca.colormap as cm


# ── Rwanda district → province lookup ────────────────────────────────────────
DISTRICT_TO_PROVINCE = {
    # Kigali City
    "Gasabo":     "Kigali City",
    "Kicukiro":   "Kigali City",
    "Nyarugenge": "Kigali City",
    # Northern Province
    "Burera":   "Northern Province",
    "Gakenke":  "Northern Province",
    "Gicumbi":  "Northern Province",
    "Musanze":  "Northern Province",
    "Rulindo":  "Northern Province",
    # Southern Province
    "Gisagara":  "Southern Province",
    "Huye":      "Southern Province",
    "Kamonyi":   "Southern Province",
    "Muhanga":   "Southern Province",
    "Nyamagabe": "Southern Province",
    "Nyanza":    "Southern Province",
    "Nyaruguru": "Southern Province",
    "Ruhango":   "Southern Province",
    # Eastern Province
    "Bugesera":  "Eastern Province",
    "Gatsibo":   "Eastern Province",
    "Kayonza":   "Eastern Province",
    "Kirehe":    "Eastern Province",
    "Ngoma":     "Eastern Province",
    "Nyagatare": "Eastern Province",
    "Rwamagana": "Eastern Province",
    # Western Province
    "Karongi":    "Western Province",
    "Ngororero":  "Western Province",
    "Nyabihu":    "Western Province",
    "Nyamasheke": "Western Province",
    "Rubavu":     "Western Province",
    "Rutsiro":    "Western Province",
    "Rusizi":     "Western Province",
}

# ── Plasma palette (shared by both colormaps) ─────────────────────────────────
PLASMA_COLORS = ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"]


def get_rwanda_map(df):
    district_counts = df["district"].value_counts().reset_index()
    district_counts.columns = ["district", "client_count"]
    count_lookup = dict(zip(district_counts["district"], district_counts["client_count"]))

    with open("dummy-data/rwanda_districts.geojson", "r", encoding="utf-8") as f:
        rwanda_geojson = json.load(f)

    for feature in rwanda_geojson["features"]:
        props = feature["properties"]
        name  = props.get("shapeName", "")
        props["district"]     = name
        props["province"]     = DISTRICT_TO_PROVINCE.get(name, "Unknown")
        props["client_count"] = int(count_lookup.get(name, 0))

    counts       = [f["properties"]["client_count"] for f in rwanda_geojson["features"]]
    min_c, max_c = min(counts), max(counts)

    colormap = cm.LinearColormap(
        colors=PLASMA_COLORS,
        vmin=min_c,
        vmax=max_c,
        caption="Number of Vehicle Clients per District",
    )

    def style_fn(feature):
        return {
            "fillColor":   colormap(feature["properties"]["client_count"]),
            "color":       "#555555",
            "weight":      1.2,
            "fillOpacity": 0.75,
        }

    def highlight_fn(feature):
        return {
            "fillColor":   colormap(feature["properties"]["client_count"]),
            "color":       "#000000",
            "weight":      2.5,
            "fillOpacity": 0.95,
        }

    m = folium.Map(
        location=[-1.94, 29.87],
        zoom_start=8,
        tiles="CartoDB positron",
    )

    tooltip = GeoJsonTooltip(
        fields=["district", "province", "client_count"],
        aliases=["District:", "Province:", "Clients:"],
        sticky=True,
        style=(
            "background-color:white;"
            "color:#1a1a2e;"
            "font-size:13px;"
            "padding:8px;"
            "border-radius:6px;"
            "border:1px solid #ccc;"
        ),
    )

    popup = GeoJsonPopup(
        fields=["district", "province", "client_count"],
        aliases=["District", "Province", "Vehicle Clients"],
    )

    GeoJson(
        rwanda_geojson,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=tooltip,
        popup=popup,
    ).add_to(m)

    step_map = cm.StepColormap(
        colors=PLASMA_COLORS,
        vmin=min_c,
        vmax=max_c,
        caption="Number of Vehicle Clients per District",
    )
    step_map.add_to(m)
    folium.LayerControl().add_to(m)
    return m._repr_html_()


def dataset_exploration(df):
    return df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False,
    )


def data_exploration(df):
    return df.describe().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
    )