"""Benchmark report generators for Markdown and standalone HTML."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_dev_toolkit.core.dataset import DatasetEvaluationResult


def generate_markdown_report(
    result: DatasetEvaluationResult,
    title: str = "AI Dev Toolkit Benchmark Report",
    min_score_threshold: float = 0.7,
) -> str:
    """Generate a GitHub Flavored Markdown benchmark report."""
    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"**Total Samples Evaluated:** `{result.total_samples}`\n")

    # Summary Statistics Table
    lines.append("## Executive Summary\n")
    lines.append("| Metric | Mean | Median | Min | Max | Sample Count |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for metric_name, summary in result.summary.items():
        name_display = metric_name.replace("_", " ").title()
        lines.append(
            f"| **{name_display}** | "
            f"`{summary.mean:.3f}` | `{summary.median:.3f}` | "
            f"`{summary.min:.3f}` | `{summary.max:.3f}` | `{summary.count}` |"
        )
    lines.append("")

    # Score Pass/Fail Rate
    if "overall" in result.summary and result.total_samples > 0:
        passing = sum(
            1 for r in result.results if r.get("overall", 0.0) >= min_score_threshold
        )
        pass_rate = (passing / result.total_samples) * 100
        pass_status = " **PASS**" if pass_rate >= 80.0 else "⚠️ **ATTENTION NEEDED**"
        lines.append("## Quality Gate Assessment\n")
        lines.append(
            f"- **Threshold Score:** `{min_score_threshold:.2f}`\n"
            f"- **Passing Samples:** `{passing}/{result.total_samples}` "
            f"({pass_rate:.1f}%)\n"
            f"- **Status:** {pass_status}\n"
        )

    # Detailed Per-Sample Table
    lines.append("## Evaluation Samples\n")
    lines.append("| ID | Overall | Prompt | Response | Key Metrics |")
    lines.append("| :--- | :---: | :--- | :--- | :--- |")

    for r in result.results[:50]:  # Cap first 50 in markdown table
        sample_id = str(r.get("id", "-"))
        overall = f"`{r.get('overall', 0.0):.2f}`"
        p_raw = r.get("prompt", "")
        r_raw = r.get("response", "")
        prompt_trunc = (p_raw[:40] + "...") if len(p_raw) > 40 else p_raw
        resp_trunc = (r_raw[:40] + "...") if len(r_raw) > 40 else r_raw

        metric_parts = []
        for k in (
            "relevance",
            "completeness",
            "faithfulness",
            "context_relevance",
            "context_recall",
            "citation_correctness",
        ):
            if k in r and r[k] is not None:
                val = r[k]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                metric_parts.append(f"{k[:3]}: {val_str}")
        metrics_str = ", ".join(metric_parts) if metric_parts else "-"

        # Escape pipe characters in table cells
        prompt_clean = prompt_trunc.replace("|", "\\|").replace("\n", " ")
        resp_clean = resp_trunc.replace("|", "\\|").replace("\n", " ")

        lines.append(
            f"| {sample_id} | {overall} | {prompt_clean} | "
            f"{resp_clean} | {metrics_str} |"
        )

    if len(result.results) > 50:
        lines.append(f"\n*(Truncated: showing 50 of {len(result.results)} samples)*\n")

    return "\n".join(lines)


def generate_html_report(
    result: DatasetEvaluationResult,
    title: str = "AI Dev Toolkit Benchmark Report",
    min_score_threshold: float = 0.7,
) -> str:
    """Generate a self-contained, responsive HTML benchmark report."""
    total = result.total_samples
    overall_mean = (
        result.summary["overall"].mean if "overall" in result.summary else 0.0
    )
    passing_count = sum(
        1 for r in result.results if r.get("overall", 0.0) >= min_score_threshold
    )
    pass_rate = (passing_count / total * 100) if total > 0 else 0.0

    # Summary rows
    summary_rows_html = ""
    for metric_name, summary in result.summary.items():
        badge_class = (
            "badge-high"
            if summary.mean >= 0.8
            else "badge-mid"
            if summary.mean >= 0.5
            else "badge-low"
        )
        escaped_name = html.escape(metric_name.replace("_", " ").title())
        summary_rows_html += f"""
        <tr>
            <td class="metric-name"><strong>{escaped_name}</strong></td>
            <td><span class="score-badge {badge_class}">{summary.mean:.3f}</span></td>
            <td>{summary.median:.3f}</td>
            <td>{summary.min:.3f}</td>
            <td>{summary.max:.3f}</td>
            <td>{summary.count}</td>
        </tr>
        """

    # Sample rows / cards
    samples_html = ""
    for idx, r in enumerate(result.results):
        sample_id = html.escape(str(r.get("id", f"sample_{idx + 1}")))
        overall = r.get("overall", 0.0)
        badge_class = (
            "badge-high"
            if overall >= 0.8
            else "badge-mid"
            if overall >= 0.5
            else "badge-low"
        )
        prompt_txt = html.escape(str(r.get("prompt", "")))
        resp_txt = html.escape(str(r.get("response", "")))
        ctx_txt = html.escape(str(r.get("context", "") or ""))
        reasoning_txt = html.escape(str(r.get("reasoning", "") or ""))

        # Per metric breakdown pills
        metric_pills = ""
        for m_key in (
            "relevance",
            "completeness",
            "faithfulness",
            "hallucination",
            "context_relevance",
            "context_recall",
            "answer_relevance",
            "citation_correctness",
        ):
            if m_key in r and r[m_key] is not None:
                val = r[m_key]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                p_class = (
                    "pill-high"
                    if isinstance(val, float) and val >= 0.8
                    else "pill-mid"
                    if isinstance(val, float) and val >= 0.5
                    else "pill-low"
                )
                metric_pills += (
                    f'<span class="pill {p_class}">'
                    f"{html.escape(m_key.replace('_', ' ').title())}: "
                    f"{val_str}</span>"
                )

        context_section = ""
        if ctx_txt:
            context_section = (
                f'<div class="sample-block"><strong>Context:</strong>'
                f"<pre>{ctx_txt}</pre></div>"
            )

        reasoning_section = ""
        if reasoning_txt:
            reasoning_section = (
                f'<div class="sample-block"><strong>Judge Reasoning:</strong>'
                f'<div class="reasoning-box">{reasoning_txt}</div></div>'
            )

        samples_html += f"""
        <div class="sample-card">
            <div class="sample-header">
                <span class="sample-id">#{sample_id}</span>
                <span class="score-badge {badge_class}">Overall: {overall:.2f}</span>
            </div>
            <div class="sample-metrics">
                {metric_pills}
            </div>
            <div class="sample-body">
                <div class="sample-block">
                    <strong>Prompt:</strong>
                    <pre>{prompt_txt}</pre>
                </div>
                <div class="sample-block">
                    <strong>Response:</strong>
                    <pre>{resp_txt}</pre>
                </div>
                {context_section}
                {reasoning_section}
            </div>
        </div>
        """

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --surface-hover: #1f242c;
            --border: #30363d;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --accent: #58a6ff;
            --success: #2ea043;
            --warning: #d29922;
            --danger: #f85149;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 32px 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 32px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
        }}
        h1 {{
            color: var(--text-heading);
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: 700;
            color: var(--text-heading);
        }}
        .stat-label {{
            font-size: 13px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 32px;
        }}
        h2 {{
            color: var(--text-heading);
            font-size: 20px;
            margin-bottom: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th, td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: #8b949e;
            font-size: 13px;
            text-transform: uppercase;
        }}
        .score-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 13px;
        }}
        .badge-high {{
            background: rgba(46, 160, 67, 0.2);
            color: #3fb950;
            border: 1px solid rgba(46, 160, 67, 0.4);
        }}
        .badge-mid {{
            background: rgba(210, 153, 34, 0.2);
            color: #e3b341;
            border: 1px solid rgba(210, 153, 34, 0.4);
        }}
        .badge-low {{
            background: rgba(248, 81, 73, 0.2);
            color: #f85149;
            border: 1px solid rgba(248, 81, 73, 0.4);
        }}

        .pill {{
            display: inline-block;
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 6px;
            margin-right: 6px;
            margin-bottom: 6px;
        }}
        .pill-high {{ background: #1f382a; color: #7ee787; }}
        .pill-mid {{ background: #3d3215; color: #f2cc60; }}
        .pill-low {{ background: #3c1e1e; color: #ffa198; }}

        .sample-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 16px;
            overflow: hidden;
        }}
        .sample-header {{
            background: rgba(255, 255, 255, 0.02);
            padding: 14px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }}
        .sample-id {{
            font-weight: 600;
            color: var(--accent);
        }}
        .sample-metrics {{
            padding: 12px 20px 4px 20px;
        }}
        .sample-body {{
            padding: 16px 20px;
        }}
        .sample-block {{
            margin-bottom: 12px;
        }}
        .sample-block strong {{
            display: block;
            font-size: 13px;
            color: #8b949e;
            margin-bottom: 4px;
        }}
        pre {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 14px;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-break: break-word;
            color: #e6edf3;
        }}
        .reasoning-box {{
            background: rgba(88, 166, 255, 0.08);
            border-left: 3px solid var(--accent);
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 13px;
            color: #d2a8ff;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{html.escape(title)}</h1>
            <p>Generated by AI Dev Toolkit (v0.3.0 RAG Evaluation Suite)</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall_mean:.3f}</div>
                <div class="stat-label">Mean Overall Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pass_rate:.1f}%</div>
                <div class="stat-label">Pass Rate (&ge; {min_score_threshold:.2f})</div>
            </div>
        </div>

        <div class="card">
            <h2>Metric Statistics Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {summary_rows_html}
                </tbody>
            </table>
        </div>

        <h2>Sample Inspections</h2>
        {samples_html}
    </div>
</body>
</html>
"""
    return html_template
