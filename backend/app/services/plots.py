import plotly.graph_objects as go

from app.models.schemas import PlotRequest


def build_plot(request: PlotRequest) -> dict:
    if request.chart_type == "scatter":
        trace = go.Scatter(x=request.x, y=request.y, mode="markers", name=request.y_label)
    elif request.chart_type == "bar":
        trace = go.Bar(x=request.x, y=request.y, name=request.y_label)
    else:
        trace = go.Scatter(x=request.x, y=request.y, mode="lines+markers", name=request.y_label)
    figure = go.Figure(data=[trace])
    figure.update_layout(
        title=request.title,
        xaxis_title=request.x_label,
        yaxis_title=request.y_label,
        template="plotly_white",
    )
    return figure.to_dict()
