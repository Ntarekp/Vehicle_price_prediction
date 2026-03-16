import pandas as pd
import json
import folium
from folium import GeoJson, GeoJsonTooltip, GeoJsonPopup
import branca.colormap as cm


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
        props["province"]     = props.get("shapeGroup", "")
        props["client_count"] = int(count_lookup.get(name, 0))

    counts       = [f["properties"]["client_count"] for f in rwanda_geojson["features"]]
    min_c, max_c = min(counts), max(counts)

    colormap = cm.LinearColormap(
        colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
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
        colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
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
