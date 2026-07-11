import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as plotly_go
import numpy as np
from plotly.subplots import make_subplots
import importlib.util

# Import required functions from other files
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

slope_mod = load_module("slope", "001_slope.py")
estimate_slope = slope_mod.estimate_slope

data_mod = load_module("data_generation", "002_data_generation.py")
generate_bouncing_ball = data_mod.generate_bouncing_ball

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Bouncing Ball Derivative Estimates"

app.layout = dbc.Container([
    dcc.Store(id="store-data"),
    dbc.Row([
        dbc.Col(html.H2("Bouncing Ball Derivative Estimates", className="text-center my-4 text-primary"), width=12)
    ]),
    dbc.Row([
        # Controls Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Generation Parameters", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Label("Duration (s)", className="text-muted"),
                    dcc.Slider(id="slider-duration", min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(2, 21, 4)}, className="mb-4"),
                    
                    html.Label("Sample Rate (Hz)", className="text-muted"),
                    dcc.Slider(id="slider-samplerate", min=100, max=2000, step=100, value=1000, marks={i: str(i) for i in range(200, 2001, 400)}, className="mb-4"),
                    
                    html.Label("Noise Level (%)", className="text-muted"),
                    dcc.Slider(id="slider-noise", min=0, max=2, step=0.05, value=1.0, marks={i: str(i) for i in range(0, 11, 2)}, className="mb-4"),
                ])
            ], className="mb-4 shadow"),
            
            dbc.Card([
                dbc.CardHeader("Estimation Parameters", className="bg-success text-white"),
                dbc.CardBody([
                    html.Label("EWMA Span (Type 2)", className="text-muted"),
                    dcc.Slider(id="slider-ewma", min=2, max=200, step=2, value=50, marks={i: str(i) for i in range(0, 201, 50)}, className="mb-4"),
                    
                    html.Label("Window Size (Type 3)", className="text-muted"),
                    dcc.Slider(id="slider-window", min=3, max=201, step=2, value=51, marks={i: str(i) for i in range(1, 202, 50)}, className="mb-4"),
                ])
            ], className="mb-4 shadow"),
            
            dbc.Card([
                dbc.CardHeader("Display Toggles", className="bg-info text-white"),
                dbc.CardBody([
                    dbc.Checklist(
                        options=[
                            {"label": "Type 1 (Simple Diff)", "value": "type1"},
                            {"label": "Type 2 (EWMA)", "value": "type2"},
                            {"label": "Type 3 (LS Conv)", "value": "type3"},
                        ],
                        value=["type1", "type2", "type3"],
                        id="checklist-toggles",
                        switch=True,
                        className="fs-5"
                    ),
                ])
            ], className="shadow"),
        ], width=12, lg=4, className="mb-4"),
        
        # Plot Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.Span("Visualization", className="text-white fw-bold"), width="auto", className="me-auto")
                    ], align="center", className="w-100 m-0"),
                    className="bg-secondary"
                ),
                dbc.CardBody([
                    dcc.Graph(id="main-plot", style={"height": "75vh"})
                ], className="p-0")
            ], className="shadow h-100")
        ], width=12, lg=8),
    ]),
], fluid=True, className="py-4 bg-dark min-vh-100")


@app.callback(
    Output("store-data", "data"),
    [
        Input("slider-duration", "value"),
        Input("slider-samplerate", "value"),
        Input("slider-noise", "value")
    ]
)
def generate_data(duration, sample_rate, noise_pct):
    data = generate_bouncing_ball(duration=duration, sample_rate=sample_rate, noise_pct=noise_pct / 100.0)
    return data.tolist()

@app.callback(
    Output("main-plot", "figure"),
    [
        Input("store-data", "data"),
        Input("slider-samplerate", "value"),
        Input("slider-ewma", "value"),
        Input("slider-window", "value"),
        Input("checklist-toggles", "value"),
    ]
)
def update_plot(data, sample_rate, ewma_span, window_size, toggles):
    if data is None:
        raise dash.exceptions.PreventUpdate

    data = np.array(data)

    # Ensure window_size is odd for Type 3 (optional but usually good)
    if window_size % 2 == 0:
        window_size += 1

    t = np.arange(len(data)) / sample_rate

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Position Trace
    fig.add_trace(
        plotly_go.Scatter(x=t, y=data, name="Position (Noisy)", line=dict(color='#00ffff', width=1.5), opacity=0.8),
        secondary_y=False,
    )

    t12 = t[1:]  # Time for type 1 and type 2
    
    if "type1" in toggles:
        slope1 = estimate_slope(data, sample_rate, estimate_type=1)
        fig.add_trace(
            plotly_go.Scatter(x=t12, y=slope1, name="Type 1 (Simple Diff)", line=dict(color='#ff9900', width=1), opacity=0.5),
            secondary_y=True,
        )
        
    if "type2" in toggles:
        slope2 = estimate_slope(data, sample_rate, estimate_type=2, ewma_span=ewma_span)
        fig.add_trace(
            plotly_go.Scatter(x=t12, y=slope2, name="Type 2 (EWMA)", line=dict(color='#33ff33', width=1.5), opacity=0.9),
            secondary_y=True,
        )
        
    if "type3" in toggles:
        try:
            slope3 = estimate_slope(data, sample_rate, estimate_type=3, window_size=window_size)
            t3_start = (window_size - 1) // 2
            t3 = t[t3_start : t3_start + len(slope3)]
            fig.add_trace(
                plotly_go.Scatter(x=t3, y=slope3, name="Type 3 (LS Conv)", line=dict(color='#ff3333', width=2.5)),
                secondary_y=True,
            )
        except Exception as e:
            pass # Handle window size error gracefully

    # Layout Updates
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        uirevision='false',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )

    fig.update_xaxes(title_text="Time (s)", showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Position (m)", secondary_y=False, color="#00ffff", showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Velocity (m/s)", secondary_y=True, color="#ffffff", showgrid=False)

    return fig

if __name__ == "__main__":
    app.run(debug=True)
