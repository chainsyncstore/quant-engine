"""Backtest report generator — produces a self-contained HTML report.

Includes equity curve (Chart.js), metrics table, trade log,
and optional model-comparison side-by-side.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from quant_v2.research.backtester import BacktestResult


def _fmt(val: float, pct: bool = False, decimals: int = 2) -> str:
    if pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:,.{decimals}f}"


def _compute_metrics(r: BacktestResult) -> dict:
    eq = r.equity_curve
    cfg = r.config
    daily = r.daily_returns

    calmar = 0.0
    if r.max_drawdown != 0.0:
        annual_return = (eq.iloc[-1] / eq.iloc[0]) ** (252 / max(len(daily), 1)) - 1
        calmar = annual_return / abs(r.max_drawdown)

    avg_fee_per_trade = r.total_fees / max(r.total_trades, 1)
    cost_drag_pct = (r.total_fees + r.total_slippage) / max(abs(r.gross_pnl), 1e-9)

    return {
        "Symbol": cfg.symbol,
        "Period": f"{cfg.start_date} → {cfg.end_date}",
        "Initial equity": _fmt(cfg.initial_equity, pct=False),
        "Final equity": _fmt(float(eq.iloc[-1]) if not eq.empty else cfg.initial_equity),
        "Net PnL": _fmt(r.net_pnl),
        "Gross PnL": _fmt(r.gross_pnl),
        "Total fees": _fmt(r.total_fees),
        "Total slippage": _fmt(r.total_slippage),
        "Cost drag": _fmt(cost_drag_pct, pct=True),
        "Sharpe ratio": _fmt(r.sharpe),
        "Max drawdown": _fmt(r.max_drawdown, pct=True),
        "Calmar ratio": _fmt(calmar),
        "Total trades": str(r.total_trades),
        "Win rate": _fmt(r.win_rate, pct=True),
        "Profit factor": _fmt(r.profit_factor),
        "Avg fee / trade": _fmt(avg_fee_per_trade),
    }


def _monthly_returns_table(daily: pd.Series) -> str:
    if daily.empty:
        return "<p>No data</p>"
    monthly = daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.to_period("M")
    rows = []
    for period, ret in monthly.items():
        color = "#c8f7c5" if ret > 0 else "#f7c8c8"
        rows.append(
            f"<tr><td>{period}</td>"
            f"<td style='background:{color};text-align:right'>{ret*100:.2f}%</td></tr>"
        )
    return (
        "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:12px'>"
        "<tr><th>Month</th><th>Return</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _trade_log_html(fills: list, max_rows: int = 200) -> str:
    if not fills:
        return "<p>No fills</p>"
    rows = []
    for f in fills[:max_rows]:
        rows.append(
            f"<tr><td>{f.timestamp}</td><td>{f.symbol}</td><td>{f.side}</td>"
            f"<td>{f.quantity:.6f}</td><td>{f.price:,.2f}</td>"
            f"<td>{f.fee_usd:.4f}</td><td>{f.slippage_usd:.4f}</td>"
            f"<td>{f.confidence:.3f}</td></tr>"
        )
    return (
        "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:11px'>"
        "<tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th>"
        "<th>Price</th><th>Fee $</th><th>Slip $</th><th>Conf</th></tr>"
        + "".join(rows)
        + f"</table><p style='font-size:11px'>Showing {min(len(fills),max_rows)} of {len(fills)} fills</p>"
    )


def _equity_chart_js(equity: pd.Series, label: str = "Equity", color: str = "#3498db") -> str:
    ts_labels = [str(t)[:16] for t in equity.index]
    values = [round(float(v), 2) for v in equity.values]
    return json.dumps({"labels": ts_labels, "values": values, "label": label, "color": color})


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report — {title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f6fa; color: #2c3e50; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin: 16px 0; }}
.metric-card {{ background: white; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
.metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }}
.metric-value {{ font-size: 20px; font-weight: bold; color: #2c3e50; margin-top: 4px; }}
.chart-container {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin: 16px 0; max-width: 1000px; }}
.comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
table {{ border-collapse: collapse; }}
th {{ background: #34495e; color: white; padding: 6px 10px; }}
td {{ padding: 4px 8px; }}
.generated {{ font-size: 11px; color: #95a5a6; margin-top: 30px; }}
</style>
</head>
<body>
<h1>Backtest Report — {title}</h1>
<p>{period}</p>

<h2>Performance Metrics</h2>
<div class="metrics-grid">
{metrics_cards}
</div>

<h2>Equity Curve</h2>
<div class="chart-container">
<canvas id="equityChart" height="80"></canvas>
</div>

{comparison_section}

<h2>Monthly Returns</h2>
{monthly_table}

<h2>Trade Log</h2>
{trade_log}

<p class="generated">Generated: {generated_at}</p>

<script>
const chartData = {chart_data};
const ctx = document.getElementById('equityChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: chartData.labels,
    datasets: [{{
      label: chartData.label,
      data: chartData.values,
      borderColor: chartData.color,
      backgroundColor: chartData.color + '22',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: true,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{ x: {{ ticks: {{ maxTicksLimit: 12, maxRotation: 45 }} }} }},
  }}
}});
{comparison_chart_js}
</script>
</body>
</html>"""

_COMPARISON_SECTION = """
<h2>Model Comparison</h2>
<div class="comparison-grid">
<div>
<h3>{label_a}</h3>
{metrics_a}
</div>
<div>
<h3>{label_b}</h3>
{metrics_b}
</div>
</div>
<div class="chart-container">
<canvas id="compChart" height="80"></canvas>
</div>
"""

_COMPARISON_CHART_JS = """
const compData = {comp_data};
const ctxComp = document.getElementById('compChart').getContext('2d');
new Chart(ctxComp, {{
  type: 'line',
  data: {{
    labels: compData[0].labels,
    datasets: compData.map((d, i) => ({{
      label: d.label,
      data: d.values,
      borderColor: d.color,
      backgroundColor: d.color + '22',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
    }}))
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{ x: {{ ticks: {{ maxTicksLimit: 12 }} }} }},
  }}
}});
"""


def _metrics_cards_html(metrics: dict) -> str:
    cards = []
    for label, val in metrics.items():
        cards.append(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{val}</div>"
            f"</div>"
        )
    return "\n".join(cards)


def _metrics_table_html(metrics: dict) -> str:
    rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        for k, v in metrics.items()
    )
    return (
        "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
        + rows + "</table>"
    )


def generate_report(
    result: BacktestResult,
    output_path: Path | None = None,
    comparison: BacktestResult | None = None,
) -> Path:
    """Generate a self-contained HTML report from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Primary backtest result.
    output_path : Path | None
        Where to write the HTML file. Defaults to ``reports/backtest_{ts}.html``.
    comparison : BacktestResult | None
        Optional second result for model comparison overlay.

    Returns
    -------
    Path
        Absolute path to the generated HTML file.
    """
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("reports") / f"backtest_{ts}.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = _compute_metrics(result)
    title = f"{result.config.symbol}"
    period = f"{result.config.start_date} → {result.config.end_date}"
    chart_data = _equity_chart_js(result.equity_curve, label=title, color="#3498db")

    comp_section = ""
    comp_chart_js = ""
    if comparison is not None:
        m_a = _compute_metrics(result)
        m_b = _compute_metrics(comparison)
        comp_section = _COMPARISON_SECTION.format(
            label_a=f"Model A ({result.config.model_version or 'active'})",
            label_b=f"Model B ({comparison.config.model_version or 'active'})",
            metrics_a=_metrics_table_html(m_a),
            metrics_b=_metrics_table_html(m_b),
        )
        comp_data = json.dumps([
            json.loads(_equity_chart_js(result.equity_curve, "Model A", "#3498db")),
            json.loads(_equity_chart_js(comparison.equity_curve, "Model B", "#e74c3c")),
        ])
        comp_chart_js = _COMPARISON_CHART_JS.format(comp_data=comp_data)

    html = _HTML_TEMPLATE.format(
        title=title,
        period=period,
        metrics_cards=_metrics_cards_html(metrics),
        chart_data=chart_data,
        comparison_section=comp_section,
        comparison_chart_js=comp_chart_js,
        monthly_table=_monthly_returns_table(result.daily_returns),
        trade_log=_trade_log_html(result.fills),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path.resolve()
