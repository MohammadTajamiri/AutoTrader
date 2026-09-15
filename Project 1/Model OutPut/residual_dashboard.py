"""
residual_dashboard.py

Interactive Plotly dashboard for diagnosing model residuals (abs_error)
across Makes, with:
  - a dropdown to switch which input column is on the X-axis
  - a dropdown to isolate a single Make (or view all at once)
  - a dashed horizontal reference line at the model's overall RMSE,
    so you can see which points are pushing error above that threshold

Usage in a notebook:
    from residual_dashboard import build_residual_dashboard
    fig = build_residual_dashboard(
        report,
        x_options=["Cylinders", "Model", "mileage", "body_type", "Year"]
    )
    fig.show()
"""

import numpy as np
import plotly.graph_objects as go


def build_residual_dashboard(df, x_options, make_col="Make", error_col="abs_error",
                               squared_error_col=None, height=650):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Results/report dataframe. Must contain make_col, error_col, and
        every column in x_options.
    x_options : list of str
        Candidate columns to switch between on the X-axis.
    make_col : str, default "Make"
    error_col : str, default "abs_error"
        Column plotted on the Y-axis.
    squared_error_col : str or None
        If you have a column of squared errors, pass it here to compute
        the true RMSE for the reference line. If None, RMSE is computed
        by squaring error_col directly (fine when error_col is already
        the raw residual/absolute error).
    height : int, default 650
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    makes = df[make_col].value_counts().index.tolist()
    default_x = x_options[0]

    # --- overall RMSE for the reference line ---
    if squared_error_col:
        rmse = np.sqrt(df[squared_error_col].mean())
    else:
        rmse = np.sqrt((df[error_col] ** 2).mean())

    # --- one trace per Make (all visible by default = "All Makes" view) ---
    fig = go.Figure()
    for make in makes:
        subset = df[df[make_col] == make]
        fig.add_trace(
            go.Scatter(
                x=subset[default_x],
                y=subset[error_col],
                mode="markers",
                marker=dict(size=6, opacity=0.6),
                name=make,
            )
        )

    # --- RMSE reference line (independent of traces, unaffected by dropdowns) ---
    fig.add_hline(
        y=rmse,
        line_dash="dash",
        line_color="red",
        annotation_text=f"RMSE = {rmse:,.0f}",
        annotation_position="top left",
    )

    n_traces = len(makes)

    # --- X-axis dropdown: restyles 'x' for every trace ---
    x_buttons = []
    for x_col in x_options:
        new_x = [df[df[make_col] == make][x_col] for make in makes]
        x_buttons.append(dict(
            label=f"X: {x_col}",
            method="restyle",
            args=[{"x": new_x}]
        ))

    # --- Make dropdown: restyles 'visible' to isolate one Make, or show all ---
    make_buttons = [dict(
        label="All Makes",
        method="restyle",
        args=[{"visible": [True] * n_traces}]
    )]
    for i, make in enumerate(makes):
        visibility = [False] * n_traces
        visibility[i] = True
        make_buttons.append(dict(
            label=make,
            method="restyle",
            args=[{"visible": visibility}]
        ))

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=x_buttons,
                direction="down",
                x=0.0, y=1.15, xanchor="left", yanchor="top",
                active=0,
            ),
            dict(
                buttons=make_buttons,
                direction="down",
                x=0.25, y=1.15, xanchor="left", yanchor="top",
                active=0,
            ),
        ],
        height=height,
        title_text=f"Residuals ({error_col}) — X: {default_x} | dashed line = overall RMSE",
        yaxis_title=error_col,
        xaxis_title=default_x,
        margin=dict(t=120),
        legend_title=make_col,
    )

    return fig