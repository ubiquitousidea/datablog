from pathlib import Path

import dash
from dash import dcc, html, Input, Output
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CSV_PATH = Path(__file__).parent / "rainbow-dataset.csv"
PHOTO_COL = "photo_base64"

# Approximate coordinates for locations logged without lat/lon
LOCATION_FALLBACK_COORDS = {"Astoria, Queens, NY": (40.7644, -73.9235)}
# Spread for markers that land on the exact same spot, in degrees
DUPLICATE_OFFSET_DEG = 0.002

df = pd.read_csv(CSV_PATH)
table_df = df.drop(columns=[PHOTO_COL])
# NaN isn't valid JSON; send empty cells to the grid as null
row_data = table_df.astype(object).where(pd.notna(table_df), None).to_dict("records")

LONG_TEXT_COLS = {"evening_conditions", "notes", "forecast_source", "next_day_rain_start_actual"}

column_defs = []
for col in table_df.columns:
    col_def = {
        "field": col,
        "filter": "agNumberColumnFilter" if pd.api.types.is_numeric_dtype(table_df[col]) else "agTextColumnFilter",
    }
    if col == "evening_date":
        col_def["pinned"] = "left"
    if col in LONG_TEXT_COLS:
        col_def.update(width=400, wrapText=True, autoHeight=True, tooltipField=col)
    column_defs.append(col_def)


def build_points(df):
    """Map coordinates per row, falling back to known locations and spreading exact duplicates."""
    fallback = df["location"].map(LOCATION_FALLBACK_COORDS)
    points = pd.DataFrame(index=df.index)
    points["approx_location"] = df["lat"].isna() & fallback.notna()
    points["map_lat"] = df["lat"].fillna(fallback.map(lambda c: c[0], na_action="ignore"))
    points["map_lon"] = df["lon"].fillna(fallback.map(lambda c: c[1], na_action="ignore"))
    points = points.dropna(subset=["map_lat", "map_lon"])

    for _, group in points.groupby(["map_lat", "map_lon"]):
        if len(group) < 2:
            continue
        angles = np.linspace(0, 2 * np.pi, len(group), endpoint=False)
        points.loc[group.index, "map_lat"] += DUPLICATE_OFFSET_DEG * np.sin(angles)
        points.loc[group.index, "map_lon"] += DUPLICATE_OFFSET_DEG * np.cos(angles)
    return points


points = build_points(df)

INTENSITY_COLORS = dict(zip(range(1, 6), px.colors.sample_colorscale("Plasma", [i / 4 for i in range(5)])))
MISSING_COLOR = "#888888"


def build_map_figure():
    intensity = df.loc[points.index, "rainbow_intensity_1_5"]
    colors = [INTENSITY_COLORS.get(int(v), MISSING_COLOR) if pd.notna(v) else MISSING_COLOR for v in intensity]
    hover = [
        f"<b>{df.at[i, 'location']}</b><br>{df.at[i, 'evening_date']}"
        + (f"<br>Intensity {int(df.at[i, 'rainbow_intensity_1_5'])}/5" if pd.notna(df.at[i, "rainbow_intensity_1_5"]) else "")
        for i in points.index
    ]

    # Fit the view to the markers
    lat_span = points["map_lat"].max() - points["map_lat"].min()
    lon_span = points["map_lon"].max() - points["map_lon"].min()
    zoom = float(np.clip(np.log2(360 / max(lat_span, lon_span, 0.01)) - 1, 1, 12))

    fig = go.Figure(go.Scattermap(
        lat=points["map_lat"],
        lon=points["map_lon"],
        mode="markers",
        marker={"size": 16, "color": colors},
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        map={
            "style": "carto-darkmatter",
            "center": {"lat": points["map_lat"].mean(), "lon": points["map_lon"].mean()},
            "zoom": zoom,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        clickmode="event",
    )
    return fig


def photo_src(b64):
    mime = "image/png" if b64.startswith("iVBOR") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def format_value(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


HIGHLIGHT_FIELDS = [
    ("rainbow_intensity_1_5", "Intensity (1–5)"),
    ("double_bow", "Double bow"),
    ("vivid_sunset", "Vivid sunset"),
    ("evening_conditions", "Evening conditions"),
    ("notes", "Notes"),
]
HEADER_FIELDS = {"location", "evening_date", "sunset_time", "photo_time_local"}
HIGHLIGHT_KEYS = {field for field, _ in HIGHLIGHT_FIELDS}


def build_info_card(idx):
    row = df.loc[idx]

    photo = row[PHOTO_COL]
    if pd.notna(photo) and photo:
        image = dbc.CardImg(src=photo_src(photo), top=True)
    else:
        image = html.Div("No photo", className="text-muted text-center py-5 border-bottom border-secondary")

    times = " · ".join(
        f"{label} {row[field]}"
        for field, label in [("photo_time_local", "photo"), ("sunset_time", "sunset")]
        if pd.notna(row[field])
    )
    header = [
        html.H4(row["location"], className="card-title mb-1"),
        html.Div(row["evening_date"], className="text-primary fw-bold"),
        html.Small(times, className="text-muted d-block"),
    ]
    if idx in points.index and points.at[idx, "approx_location"]:
        header.append(dbc.Badge("approximate location", color="warning", className="mt-2"))

    highlights = [
        html.Div([html.Small(label, className="text-muted d-block"), html.Div(format_value(row[field]))], className="mb-2")
        for field, label in HIGHLIGHT_FIELDS
        if pd.notna(row[field])
    ]

    details = [
        html.Tr([html.Td(col, className="text-muted"), html.Td(format_value(row[col]))])
        for col in table_df.columns
        if col not in HEADER_FIELDS | HIGHLIGHT_KEYS and pd.notna(row[col])
    ]

    return dbc.Card([
        image,
        dbc.CardBody([
            html.Div(header, className="mb-3"),
            html.Div(highlights, className="mb-3"),
            dbc.Table(html.Tbody(details), size="sm", bordered=False, className="small mb-0"),
        ]),
    ], className="shadow")


placeholder_card = dbc.Card(
    dbc.CardBody(html.P("Click a rainbow on the map", className="text-muted text-center my-5")),
    className="shadow",
)

map_tab = dbc.Row([
    dbc.Col(
        dcc.Graph(id="rainbow-map", figure=build_map_figure(), config={"scrollZoom": True}, style={"height": "75vh"}),
        md=7,
    ),
    dbc.Col(
        html.Div(placeholder_card, id="rainbow-card", style={"height": "75vh", "overflowY": "auto"}),
        md=5,
    ),
], className="pt-3")

table_tab = dbc.Card([
    dbc.CardHeader(
        dbc.Row([
            dbc.Col(html.Span(f"{len(table_df)} rows × {len(table_df.columns)} columns", className="text-white fw-bold"), width="auto", className="me-auto"),
            dbc.Col(dbc.Button("Export CSV", id="btn-export", size="sm", color="primary"), width="auto"),
        ], align="center", className="w-100 m-0"),
        className="bg-secondary"
    ),
    dbc.CardBody([
        dag.AgGrid(
            id="rainbow-grid",
            rowData=row_data,
            columnDefs=column_defs,
            defaultColDef={"sortable": True, "filter": True, "floatingFilter": True, "resizable": True},
            dashGridOptions={
                "theme": {"function": "themeAlpine.withPart(agGrid.colorSchemeDark)"},
                "pagination": True,
                "paginationPageSize": 20,
                "tooltipShowDelay": 300,
            },
            csvExportParams={"fileName": "rainbow-dataset.csv"},
            style={"height": "75vh"},
        )
    ], className="p-0")
], className="shadow mt-3")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Rainbow Dataset"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Rainbow Dataset", className="text-center my-4 text-primary"), width=12)
    ]),
    dbc.Tabs([
        dbc.Tab(map_tab, label="Map", tab_id="tab-map"),
        dbc.Tab(table_tab, label="Table", tab_id="tab-table"),
    ], active_tab="tab-map"),
], fluid=True, className="py-4 bg-dark min-vh-100")


@app.callback(
    Output("rainbow-card", "children"),
    Input("rainbow-map", "clickData"),
    prevent_initial_call=True,
)
def show_rainbow(click_data):
    # pointNumber is the marker's position in the trace, which follows points.index
    return build_info_card(points.index[click_data["points"][0]["pointNumber"]])


@app.callback(
    Output("rainbow-grid", "exportDataAsCsv"),
    Input("btn-export", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(n_clicks):
    return True


if __name__ == "__main__":
    app.run(debug=True)
