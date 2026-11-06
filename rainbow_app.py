from pathlib import Path

import dash
from dash import html, Input, Output
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd

CSV_PATH = Path(__file__).parent / "rainbow-dataset.csv"

df = pd.read_csv(CSV_PATH)
# NaN isn't valid JSON; send empty cells to the grid as null
row_data = df.astype(object).where(pd.notna(df), None).to_dict("records")

LONG_TEXT_COLS = {"evening_conditions", "notes", "forecast_source", "next_day_rain_start_actual"}

column_defs = []
for col in df.columns:
    col_def = {
        "field": col,
        "filter": "agNumberColumnFilter" if pd.api.types.is_numeric_dtype(df[col]) else "agTextColumnFilter",
    }
    if col == "evening_date":
        col_def["pinned"] = "left"
    if col in LONG_TEXT_COLS:
        col_def.update(width=400, wrapText=True, autoHeight=True, tooltipField=col)
    column_defs.append(col_def)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Rainbow Dataset"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Rainbow Dataset", className="text-center my-4 text-primary"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.Span(f"{len(df)} rows × {len(df.columns)} columns", className="text-white fw-bold"), width="auto", className="me-auto"),
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
                        style={"height": "80vh"},
                    )
                ], className="p-0")
            ], className="shadow")
        ], width=12),
    ]),
], fluid=True, className="py-4 bg-dark min-vh-100")


@app.callback(
    Output("rainbow-grid", "exportDataAsCsv"),
    Input("btn-export", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(n_clicks):
    return True


if __name__ == "__main__":
    app.run(debug=True)
