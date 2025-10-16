"""
plot_utils.py
-------------
Plotly helper to render quantum tunneling zones to a standalone HTML.
Zero-dependency bloat: just Plotly + Pandas.
"""
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import os

# Zone color mapping
COLORS = {"LOW": "#4C78A8", "MEDIUM": "#F58518", "HIGH": "#E45756"}

# Thresholds (can be overridden via env vars)
THRESH_LOW = float(os.getenv("QT_LOW", "0.25"))
THRESH_HIGH = float(os.getenv("QT_HIGH", "0.50"))


def render_tunneling_html(df: pd.DataFrame,
                          html_path: Path,
                          title: str = "Quantum Tunneling Zones",
                          kappa: float = 500.0):
    """
    Render quantum tunneling zones as a standalone HTML with two charts.
    
    Required columns: strike_k, tunnel_prob, tunnel_flag
    Optional columns: opt_side, volume_x, bid, ask, spread, iv_x, oi_x, gamma_x, mid
    
    Args:
        df: DataFrame with tunneling metrics
        html_path: Output path for HTML file
        title: Chart title
        kappa: Kappa value used in calculation (for display)
    """
    try:
        d = df.copy()
    except Exception as e:
        print(f"Error copying dataframe: {e}")
        raise
    
    # Guard rails
    if "strike_k" not in d or "tunnel_prob" not in d or "tunnel_flag" not in d:
        raise ValueError("Required columns (strike_k, tunnel_prob, tunnel_flag) missing for plotting.")
    
    d = d.sort_values("strike_k")
    d["flag_color"] = d["tunnel_flag"].map(COLORS).fillna("#6B6B6B")
    
    # Check for optional columns
    has_side = "opt_side" in d.columns and d["opt_side"].notna().any()
    
    # Helper to safely get column values with fallback
    def safe_col(col_name, default=0.0):
        if col_name in d.columns:
            vals = pd.to_numeric(d[col_name], errors='coerce').fillna(default)
            return vals.values
        return [default] * len(d)
    
    # Reset index to ensure clean 0-based indexing
    d = d.reset_index(drop=True)
    
    # Build customdata array with available columns - ensure all are numeric arrays
    try:
        customdata_list = []
        for idx, row in d.iterrows():
            row_data = [
                str(row.get("tunnel_flag", "N/A")),
                float(row.get("mid", 0) if pd.notna(row.get("mid")) else 0),
                float(row.get("bid", 0) if pd.notna(row.get("bid")) else 0),
                float(row.get("ask", 0) if pd.notna(row.get("ask")) else 0),
                float(row.get("spread", 0) if pd.notna(row.get("spread")) else 0),
                float(row.get("iv_x", 0) if pd.notna(row.get("iv_x")) else 0),
                float(row.get("oi_x", 0) if pd.notna(row.get("oi_x")) else 0),
                float(row.get("volume_x", 0) if pd.notna(row.get("volume_x")) else 0),
            ]
            customdata_list.append(row_data)
    except Exception as e:
        print(f"Warning: Error building customdata: {e}")
        customdata_list = [[str(f), 0, 0, 0, 0, 0, 0, 0] for f in d["tunnel_flag"]]
    
    # Build stacked bar chart by opt_side
    try:
        traces = []
        sides = ["C", "P"] if has_side else [None]
        
        for side in sides:
            if side is None:
                sub = d.copy()
            else:
                sub = d[d["opt_side"] == side].copy()
            
            if len(sub) == 0:
                continue
            
            # Build customdata for this subset using iloc
            sub_custom = []
            for idx in sub.index:
                sub_custom.append(customdata_list[idx])
            
            traces.append(go.Bar(
                x=sub["strike_k"].tolist(),
                y=sub["tunnel_prob"].tolist(),
                marker_color=sub["flag_color"].tolist(),
                name=("Calls" if side == "C" else "Puts") if side else "All",
                hovertemplate=(
                    "<b>Strike: %{x}</b><br>"
                    "Probability: %{y:.3f}<br>"
                    "Zone: %{customdata[0]}<br>"
                    "─────────────────<br>"
                    "Mid: %{customdata[1]:.2f}<br>"
                    "Bid: %{customdata[2]:.2f} | Ask: %{customdata[3]:.2f}<br>"
                    "Spread: %{customdata[4]:.3f}<br>"
                    "IV: %{customdata[5]:.3f}<br>"
                    "OI: %{customdata[6]:.0f} | Vol: %{customdata[7]:.0f}"
                    "<extra></extra>"
                ),
                customdata=sub_custom
            ))
        
        # Create bar chart
        fig_bar = go.Figure(data=traces)
        fig_bar.update_layout(
            title={"text": title + " — Probability by Strike", "font": {"size": 20}},
            xaxis_title="Strike Price",
            yaxis_title="Tunneling Probability (0–1)",
            barmode="stack" if has_side else "relative",
            template="plotly_white",
            legend={"title": "Side", "orientation": "h", "y": 1.1, "x": 0.5, "xanchor": "center"},
            height=500,
            hovermode="closest"
        )
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Add threshold lines
    fig_bar.add_hline(y=THRESH_LOW, line_dash="dash", line_color="gray", 
                      annotation_text=f"Low Threshold ({THRESH_LOW})", 
                      annotation_position="right")
    fig_bar.add_hline(y=THRESH_HIGH, line_dash="dash", line_color="gray",
                      annotation_text=f"High Threshold ({THRESH_HIGH})",
                      annotation_position="right")
    
    # Create scatter plot (sized by volume if available)
    try:
        has_volume = "volume_x" in d.columns and d["volume_x"].notna().any()
        
        # Simplified approach - use go.Scatter directly to avoid plotly express dict issues
        scatter_traces = []
        
        for flag, color in COLORS.items():
            flag_data = d[d["tunnel_flag"] == flag]
            if flag_data.empty:
                continue
            
            # Build hover text manually
            hover_texts = []
            for _, row in flag_data.iterrows():
                hover_parts = [
                    f"<b>Strike: {row['strike_k']:.0f}</b>",
                    f"Probability: {row['tunnel_prob']:.3f}",
                    f"Zone: {row['tunnel_flag']}"
                ]
                if "mid" in row.index and pd.notna(row["mid"]):
                    hover_parts.append(f"Mid: {row['mid']:.2f}")
                if "spread" in row.index and pd.notna(row["spread"]):
                    hover_parts.append(f"Spread: {row['spread']:.3f}")
                if "iv_x" in row.index and pd.notna(row["iv_x"]):
                    hover_parts.append(f"IV: {row['iv_x']:.3f}")
                if "volume_x" in row.index and pd.notna(row["volume_x"]):
                    hover_parts.append(f"Volume: {row['volume_x']:.0f}")
                if "oi_x" in row.index and pd.notna(row["oi_x"]):
                    hover_parts.append(f"OI: {row['oi_x']:.0f}")
                
                hover_texts.append("<br>".join(hover_parts))
            
            # Build marker configuration
            marker_config = {"color": color}
            if has_volume:
                # Add size based on volume
                sizes = flag_data["volume_x"].fillna(1).clip(lower=1).tolist()
                max_size = max(sizes) if sizes else 1
                marker_config["size"] = sizes
                marker_config["sizemode"] = "diameter"
                marker_config["sizeref"] = 2.0 * max_size / (20.0 ** 2) if max_size > 0 else 1
            else:
                marker_config["size"] = 8
            
            scatter_traces.append(go.Scatter(
                x=flag_data["strike_k"].tolist(),
                y=flag_data["tunnel_prob"].tolist(),
                mode="markers",
                name=flag,
                marker=marker_config,
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>"
            ))
        
        fig_scatter = go.Figure(data=scatter_traces)
        
        fig_scatter.update_layout(
            title={"text": "Distribution & Volume Weighting", "font": {"size": 20}},
            xaxis_title="Strike Price",
            yaxis_title="Tunneling Probability (0–1)",
            template="plotly_white",
            legend={"title": "Zone", "orientation": "h", "y": 1.1, "x": 0.5, "xanchor": "center"},
            height=500,
            hovermode="closest"
        )
        
        # Add threshold lines to scatter
        fig_scatter.add_hline(y=THRESH_LOW, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_hline(y=THRESH_HIGH, line_dash="dash", line_color="gray", opacity=0.5)
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Generate standalone HTML
    try:
        print("Converting bar chart to HTML...")
        bar_html = fig_bar.to_html(
            full_html=False, 
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "displaylogo": False}
        )
    except Exception as e:
        print(f"Error converting bar chart to HTML: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    try:
        print("Converting scatter chart to HTML...")
        scatter_html = fig_scatter.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": True, "displaylogo": False}
        )
    except Exception as e:
        print(f"Error converting scatter chart to HTML: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<style>
body {{
    max-width: 1400px;
    margin: 24px auto;
    padding: 0 20px;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #fafafa;
    color: #333;
}}
.header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}}
.header h1 {{
    margin: 0 0 8px 0;
    font-size: 32px;
    font-weight: 700;
}}
.info {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 24px;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}}
.info h3 {{
    margin: 0 0 12px 0;
    color: #667eea;
    font-size: 18px;
}}
.zone-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: 600;
    margin: 0 8px 8px 0;
}}
.zone-low {{ background-color: {COLORS['LOW']}; color: white; }}
.zone-medium {{ background-color: {COLORS['MEDIUM']}; color: white; }}
.zone-high {{ background-color: {COLORS['HIGH']}; color: white; }}
.chart-container {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 24px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}}
.spacer {{
    height: 32px;
}}
code {{
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "Courier New", monospace;
    font-size: 14px;
}}
</style>
</head>
<body>
<div class="header">
    <h1>🌊 {title}</h1>
    <p style="margin:0;opacity:0.95;">Quantum tunneling probability analysis for option strikes</p>
</div>

<div class="info">
    <h3>📊 Zone Classification</h3>
    <div>
        <span class="zone-badge zone-low">LOW</span> 
        <span style="color:#666;">Probability &lt; {THRESH_LOW:.2f}</span>
    </div>
    <div>
        <span class="zone-badge zone-medium">MEDIUM</span>
        <span style="color:#666;">{THRESH_LOW:.2f} ≤ Probability &lt; {THRESH_HIGH:.2f}</span>
    </div>
    <div>
        <span class="zone-badge zone-high">HIGH</span>
        <span style="color:#666;">Probability ≥ {THRESH_HIGH:.2f}</span>
    </div>
    <p style="margin:16px 0 0 0;color:#666;font-size:14px;">
        <strong>Model parameters:</strong> κ (kappa) = {kappa:.1f} | 
        Barrier strength scaled by IV compression | 
        Imbalance from call/put volume differentials
    </p>
</div>

<div class="chart-container">
    {bar_html}
</div>

<div class="spacer"></div>

<div class="chart-container">
    {scatter_html}
</div>

<div class="info" style="margin-top:32px;border-left-color:#764ba2;">
    <h3>ℹ️ About Quantum Tunneling Model</h3>
    <p style="margin:8px 0;line-height:1.6;color:#555;">
        This model calculates the probability of price "tunneling" through resistance/support barriers 
        based on option market microstructure. Higher probabilities indicate weaker barriers due to:
    </p>
    <ul style="margin:8px 0;line-height:1.8;color:#555;">
        <li><strong>Wide bid-ask spreads</strong> → less conviction at that strike</li>
        <li><strong>Low volume/open interest</strong> → thin order book</li>
        <li><strong>Order imbalance</strong> → directional pressure from calls vs puts</li>
        <li><strong>IV compression</strong> → market uncertainty reduced</li>
    </ul>
    <p style="margin:8px 0 0 0;color:#666;font-size:13px;">
        Generated using <code>quantum_tunneling_model.py</code> | 
        View source data: <code>tunneling_report.csv</code>
    </p>
</div>

</body>
</html>"""
    
    # Write to file
    html_path.write_text(html_content, encoding="utf-8")
    return html_path

