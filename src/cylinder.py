import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import pyarrow.parquet as pq

#from ydata_profiling import ProfileReport
import pandas as pd

# Read as a Table
table = pq.read_table('Grassi/data/training/events.parquet')
truth = pq.read_table('Grassi/data/training/truth.parquet')

# Convert to Pandas only if/when needed
df = table.to_pandas()
df_truth = truth.to_pandas()

df_total = pd.concat([df,df_truth], axis=1)

# ── 1. Example data ──────────────────────────────────────────────────────────
print(df_total.head())
df_total['r']= np.sqrt(df_total['x']**2 + df_total['y']**2)
df_total['phi'] = np.arctan2(df_total['y'], df_total['x'])
print(df_total.head())
print(len(df_total['label_name'].unique()))
print(len(df_total['label_name'].unique()))
print(len(df_total['label_name'].unique()))
df = df_total.sample(frac=0.1, random_state=42).reset_index(drop=True) 

# ── 2. Derived limits ────────────────────────────────────────────────────────
R     = df["r"].max()
z_min = df["z"].min()
z_max = df["z"].max()

# ── 3. Convert cylindrical → Cartesian ───────────────────────────────────────
df["x"] = df["r"] * np.cos(df["phi"])
df["y"] = df["r"] * np.sin(df["phi"])

# ── 4. Build discrete color map ───────────────────────────────────────────────
categories  = df["label_name"].unique().tolist()
palette     = px.colors.qualitative.Plotly   # swap for Bold, Vivid, Safe, etc.
color_map   = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

# ── 5. Cylinder surface helpers ───────────────────────────────────────────────
def make_wall(R, z_min, z_max, n_theta=120):
    th = np.linspace(0, 2 * np.pi, n_theta)
    z  = np.array([z_min, z_max])
    TH, Z = np.meshgrid(th, z)
    return R * np.cos(TH), R * np.sin(TH), Z

def make_disk(z_level, R, n_r=30, n_theta=60):
    r  = np.linspace(0, R, n_r)
    th = np.linspace(0, 2 * np.pi, n_theta)
    R_d, TH = np.meshgrid(r, th)
    return R_d * np.cos(TH), R_d * np.sin(TH), np.full_like(R_d, z_level)

surface_kwargs = dict(
    colorscale=[[0, "rgba(100,160,255,0.12)"], [1, "rgba(100,160,255,0.12)"]],
    showscale=False, opacity=0.18, hoverinfo="skip",
    lighting=dict(ambient=1),
)

Xw, Yw, Zw = make_wall(R, z_min, z_max)
Xb, Yb, Zb = make_disk(z_min, R)
Xt, Yt, Zt = make_disk(z_max, R)

cylinder_wall = go.Surface(x=Xw, y=Yw, z=Zw,
                            surfacecolor=np.zeros_like(Zw), **surface_kwargs)
bottom_cap    = go.Surface(x=Xb, y=Yb, z=Zb,
                            surfacecolor=np.zeros_like(Zb), **surface_kwargs)
top_cap       = go.Surface(x=Xt, y=Yt, z=Zt,
                            surfacecolor=np.zeros_like(Zt), **surface_kwargs)

# ── 6. One Scatter3d trace per category (gives a clean legend) ───────────────
scatter_traces = []
for cat in categories:
    mask = df["label_name"] == cat
    sub  = df[mask]
    scatter_traces.append(go.Scatter3d(
        x=sub["x"], y=sub["y"], z=sub["z"],
        mode="markers",
        name=str(cat),
        marker=dict(
            size=4,
            color=color_map[cat],
            opacity=0.85,
            line=dict(width=0),
        ),
        hovertemplate=(
            f"<b>{cat}</b><br>"
            "r: %{customdata[0]:.3f}<br>"
            "θ: %{customdata[1]:.3f} rad<br>"
            "z: %{z:.3f}<extra></extra>"
        ),
        customdata=np.stack([sub["r"], sub["phi"]], axis=1),
    ))

# ── 7. Assemble figure ────────────────────────────────────────────────────────
fig = go.Figure(data=[cylinder_wall, bottom_cap, top_cap] + scatter_traces)

fig.update_layout(
    title=dict(text="Data points inside a cylinder — categorical hue", x=0.5),
    scene=dict(
        xaxis=dict(title="X", range=[-R * 1.15, R * 1.15]),
        yaxis=dict(title="Y", range=[-R * 1.15, R * 1.15]),
        zaxis=dict(title="Z", range=[z_min - 0.5, z_max + 0.5]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1.5),
        bgcolor="rgba(10,12,20,1)",
        xaxis_gridcolor="rgba(255,255,255,0.07)",
        yaxis_gridcolor="rgba(255,255,255,0.07)",
        zaxis_gridcolor="rgba(255,255,255,0.07)",
    ),
    paper_bgcolor="rgba(10,12,20,1)",
    font_color="white",
    margin=dict(l=0, r=0, t=50, b=0),
    legend=dict(title="Category", x=0.01, y=0.99),
)

#fig.show()
fig.write_html("cylinder_categorical.html")