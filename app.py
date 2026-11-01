"""FlashCache-KV | Interactive performance and telemetry dashboard.

This dashboard is a presentation layer around the existing FlashCache-KV
runtime. It does not replace or reimplement the cache algorithms.
"""

from __future__ import annotations

import html
import math
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from flashcache import FlashCacheRuntime, MemoryMonitor
from flashcache.compression import KIVIDequantizer, KIVIQuantizer


# -----------------------------------------------------------------------------
# Page configuration and visual system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FlashCache-KV | KV Cache Optimizer",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#17213B"
MUTED = "#64718B"
PURPLE = "#5B3CE6"
PURPLE_LIGHT = "#EEEAFE"
BLUE = "#3985E8"
GREEN = "#16834A"
GREEN_LIGHT = "#EAF7EF"
AMBER = "#B7791F"
BORDER = "#DEE4F0"
SURFACE = "#FFFFFF"
CANVAS = "#F6F7FB"
SLATE = "#9AA5B5"

st.markdown(
    f"""
    <style>
      :root {{
        --fc-bg: {CANVAS};
        --fc-surface: {SURFACE};
        --fc-text: {NAVY};
        --fc-muted: {MUTED};
        --fc-border: {BORDER};
        --fc-accent: {PURPLE};
      }}
      html, body, [data-testid="stAppViewContainer"] {{
        background: var(--fc-bg) !important;
        color: var(--fc-text) !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      [data-testid="stHeader"] {{
        background: rgba(246,247,251,.96) !important;
        border-bottom: 1px solid rgba(222,228,240,.55);
      }}
      [data-testid="stToolbar"] {{ right: 1rem; }}

      /* Let Streamlit control the sidebar width and animation. Fixed min/max
         widths break the native collapse behaviour and prevent the main pane
         from reclaiming the available viewport. */
      [data-testid="stSidebar"] {{
        background: #FFFFFF !important;
        border-right: 1px solid var(--fc-border);
      }}
      [data-testid="stSidebar"] > div:first-child {{ padding-top: 1rem; }}
      [data-testid="stSidebarContent"] {{ padding: 1rem .9rem 1rem .9rem; }}

      /* The main content must use the complete width left by the sidebar.
         In particular, do not cap it at a fixed max-width: that leaves a blank
         strip after the sidebar is collapsed. */
      [data-testid="stMainBlockContainer"],
      .stMainBlockContainer,
      .block-container {{
        width: 100% !important;
        max-width: none !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding-top: 2.15rem !important;
        padding-right: clamp(1rem, 2.1vw, 2.35rem) !important;
        padding-bottom: 2.6rem !important;
        padding-left: clamp(1rem, 2.1vw, 2.35rem) !important;
        box-sizing: border-box !important;
      }}

      h1, h2, h3, h4 {{ color: var(--fc-text); letter-spacing: -0.025em; }}
      h1 {{ font-size: 2rem !important; font-weight: 760 !important; }}
      h2 {{ font-size: 1.38rem !important; font-weight: 730 !important; }}
      h3 {{ font-size: 1.08rem !important; font-weight: 700 !important; }}
      p, [data-testid="stCaptionContainer"] {{ color: var(--fc-muted); }}
      [data-testid="stMarkdownContainer"] p {{ margin-bottom: .45rem; }}
      [data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--fc-surface) !important;
        border: 1px solid var(--fc-border) !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 8px rgba(24,38,70,.035);
      }}
      [data-testid="stMetric"] {{
        background: #FFFFFF; border: 1px solid var(--fc-border);
        border-radius: 13px; padding: 14px 16px;
      }}
      [data-testid="stMetricLabel"] {{ color: var(--fc-muted) !important; font-size: .84rem !important; }}
      [data-testid="stMetricValue"] {{ color: var(--fc-text) !important; font-weight: 760 !important; }}
      [data-testid="stButton"] > button,
      [data-testid="stFormSubmitButton"] > button,
      [data-testid="stBaseButton-secondary"] {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100%;
        min-height: 46px !important;
        padding: .68rem 1.05rem !important;
        border-radius: 10px !important;
        font-size: .9rem !important;
        line-height: 1.25 !important;
        font-weight: 680 !important;
        border: 1px solid #D4DBE9 !important;
        color: var(--fc-text) !important;
        -webkit-text-fill-color: var(--fc-text) !important;
        background: #FFFFFF !important;
        white-space: normal !important;
        overflow: visible !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.035);
        transition: background .15s ease, border-color .15s ease, transform .15s ease;
      }}
      [data-testid="stButton"] > button:hover,
      [data-testid="stFormSubmitButton"] > button:hover,
      [data-testid="stBaseButton-secondary"]:hover {{
        border-color: #B8C3D8 !important;
        background: #F7F8FC !important;
      }}

      /* Streamlit's current DOM uses data-testid on the button element;
         older releases used the kind attribute. Target both, including child
         spans, so primary-button text never inherits a dark foreground. */
      [data-testid="stBaseButton-primary"],
      [data-testid="stButton"] > button[kind="primary"],
      [data-testid="stFormSubmitButton"] > button[kind="primary"] {{
        background: var(--fc-accent) !important;
        border: 1px solid var(--fc-accent) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        min-height: 48px !important;
        font-weight: 730 !important;
        box-shadow: 0 5px 14px rgba(91,60,230,.16) !important;
      }}
      [data-testid="stBaseButton-primary"] *,
      [data-testid="stButton"] > button[kind="primary"] *,
      [data-testid="stFormSubmitButton"] > button[kind="primary"] * {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
      }}
      [data-testid="stBaseButton-primary"]:hover,
      [data-testid="stButton"] > button[kind="primary"]:hover {{
        background: #4D30D1 !important;
        border-color: #4D30D1 !important;
        transform: translateY(-1px);
      }}
      [data-testid="stBaseButton-primary"]:focus-visible,
      [data-testid="stButton"] > button[kind="primary"]:focus-visible {{
        outline: 3px solid rgba(91,60,230,.22) !important;
        outline-offset: 2px !important;
      }}
      [data-testid="stSlider"] [role="slider"] {{ background: var(--fc-accent) !important; }}
      [data-testid="stSlider"] [data-testid="stTickBar"] {{ background: #D9DDEA !important; }}
      [data-baseweb="select"] > div {{
        background: #FFFFFF !important; color: var(--fc-text) !important;
        border-color: #D5DCE9 !important; border-radius: 9px !important;
      }}
      [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {{
        background: #FFFFFF !important; color: var(--fc-text) !important;
      }}
      [data-testid="stDataFrame"] {{ border: 1px solid var(--fc-border); border-radius: 10px; overflow: hidden; }}
      [data-testid="stPlotlyChart"] {{ background: transparent !important; }}
      [data-testid="stAlert"] {{ border-radius: 11px !important; }}
      [data-testid="stRadio"] label {{ color: var(--fc-text) !important; }}
      [data-testid="stSidebar"] [data-testid="stRadio"] > div {{ gap: .18rem; }}
      [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        padding: .53rem .62rem; border-radius: 9px; font-weight: 560;
      }}
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        background: #EEEAFE; color: #3620A6 !important;
      }}
      [data-testid="stExpander"] {{ border: 1px solid var(--fc-border) !important; border-radius: 10px !important; }}
      .fc-brand {{ display:flex; align-items:center; gap:10px; padding: .1rem .2rem .8rem .2rem; }}
      .fc-logo {{ width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center;
        color:#FFFFFF; background:linear-gradient(140deg,#6C4BF2,#4728CC); font-size:20px; font-weight:800; }}
      .fc-brand-title {{ font-size:1.18rem; line-height:1.12; color:var(--fc-text); font-weight:780; letter-spacing:-.04em; }}
      .fc-brand-sub {{ color:var(--fc-muted); font-size:.66rem; letter-spacing:.12em; margin-top:3px; }}
      .fc-side-label {{ color:#8791A5; font-weight:750; letter-spacing:.14em; font-size:.67rem; margin:1.1rem 0 .35rem .3rem; }}
      .fc-system {{ border:1px solid var(--fc-border); border-radius:13px; background:#FAFBFD; padding:13px 13px 11px 13px; margin-top:1rem; }}
      .fc-system-title {{ font-weight:750; color:var(--fc-text); margin-bottom:9px; font-size:.91rem; }}
      .fc-system-row {{ display:flex; align-items:center; justify-content:space-between; gap:8px; padding:5px 0; font-size:.78rem; }}
      .fc-system-row span:first-child {{ color:var(--fc-muted); }}
      .fc-system-row strong {{ color:var(--fc-text); font-weight:650; text-align:right; }}
      .fc-ready {{ display:flex; align-items:center; gap:7px; color:{GREEN}; font-size:.78rem; font-weight:700; margin-top:7px; }}
      .fc-dot {{ width:8px; height:8px; border-radius:50%; background:#20A35A; display:inline-block; }}
      .fc-page-head {{
        display:flex; justify-content:space-between; align-items:center; gap:1.25rem;
        width:100%; min-height:76px; padding:.4rem 0 .55rem; margin:.2rem 0 1.25rem;
        overflow:visible;
      }}
      .fc-page-head-main {{ min-width:0; flex:1 1 auto; }}
      .fc-page-title {{ font-size:clamp(1.72rem,2.3vw,2.12rem); line-height:1.15; font-weight:790; letter-spacing:-.045em; color:var(--fc-text); }}
      .fc-page-sub {{ color:var(--fc-muted); font-size:.94rem; line-height:1.48; margin-top:.35rem; max-width:900px; }}
      .fc-badges {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; align-items:center; padding:.45rem 0; }}
      .fc-header-badge-wrap {{ min-width:205px; display:flex; justify-content:flex-end; align-items:center; }}
      .fc-header-rule {{ height:1px; background:linear-gradient(90deg, #E4E8F0 0%, rgba(228,232,240,0) 100%); margin:.2rem 0 1.15rem; }}
      .fc-badge {{ display:inline-flex; align-items:center; gap:7px; border:1px solid var(--fc-border); background:#FFFFFF;
        color:#42506A; border-radius:999px; padding:7px 11px; font-size:.75rem; font-weight:650; white-space:nowrap; }}
      .fc-badge-green {{ color:#146B3A; background:#EFF9F2; border-color:#CDEBD7; }}
      .fc-section-title {{ font-size:1.12rem; font-weight:760; color:var(--fc-text); letter-spacing:-.025em; }}
      .fc-section-sub {{ color:var(--fc-muted); font-size:.81rem; margin-top:2px; }}
      .fc-kpi {{ height:100%; background:#FFFFFF; border:1px solid var(--fc-border); border-radius:13px; padding:14px 15px; min-height:102px; }}
      .fc-kpi-label {{ color:var(--fc-muted); font-size:.8rem; font-weight:600; margin-bottom:8px; }}
      .fc-kpi-value {{ color:var(--fc-text); font-size:clamp(1.25rem,2vw,1.7rem); line-height:1.15; font-weight:790; letter-spacing:-.04em; overflow-wrap:anywhere; }}
      .fc-kpi-note {{ color:#8993A6; font-size:.7rem; margin-top:6px; }}
      .fc-status {{ display:flex; align-items:center; flex-wrap:wrap; gap:9px; border:1px solid #CBE8D4; background:#EFF9F2; color:#146B3A; border-radius:11px; padding:11px 14px; font-size:.84rem; font-weight:650; margin:.4rem 0 .8rem 0; }}
      .fc-stage-num {{ width:27px; height:27px; display:flex; align-items:center; justify-content:center; border-radius:50%; background:#EEEAFE; color:#4C35C6; font-weight:750; font-size:.8rem; }}
      .fc-stage-title {{ font-size:.85rem; font-weight:720; color:var(--fc-text); }}
      .fc-stage-desc {{ color:var(--fc-muted); font-size:.73rem; margin-top:2px; line-height:1.35; }}
      .fc-pill {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#EAF7EF; color:#14723D; font-size:.68rem; font-weight:700; white-space:nowrap; }}
      .fc-muted-pill {{ background:#F0F2F6; color:#667085; }}
      .fc-note {{ font-size:.76rem; line-height:1.45; color:var(--fc-muted); }}
      .fc-empty {{ border:1px dashed #CBD3E2; background:#FBFCFE; border-radius:13px; padding:24px; text-align:center; color:var(--fc-muted); }}
      .fc-small-label {{ font-size:.7rem; color:#8993A6; text-transform:uppercase; letter-spacing:.11em; font-weight:750; }}
      .fc-detail-value {{ color:var(--fc-text); font-size:.9rem; font-weight:700; margin-top:4px; }}
      @media (max-width: 1100px) {{
        .fc-page-head {{ align-items:flex-start; flex-direction:column; gap:.45rem; }}
        .fc-badges {{ justify-content:flex-start; padding:.1rem 0 .25rem; }}
        .fc-header-badge-wrap {{ min-width:0; justify-content:flex-start; }}
      }}
      @media (max-width: 760px) {{
        [data-testid="stMainBlockContainer"], .stMainBlockContainer, .block-container {{
          padding-left: .8rem !important; padding-right: .8rem !important;
          padding-top: 1.25rem !important;
        }}
        .fc-page-title {{ font-size:1.55rem; }}
        .fc-kpi {{ padding:11px 12px; min-height:90px; }}
        .fc-kpi-value {{ font-size:1.22rem; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Historical reference data from the original dashboard source.
# These are reference records, not measurements from the current local run.
# -----------------------------------------------------------------------------
HISTORICAL_BENCHMARKS = [
    {
        "strategy": "Full Cache (Baseline)",
        "type": "Full Precision",
        "capacity": 222,
        "sink_tokens": 0,
        "retained_tokens": 222,
        "kv_mb": 2.6016,
        "kv_bytes": 2727936,
        "compression_pct": 0.0,
        "compression_ratio": 1.0,
        "latency_sec": 4.848,
        "throughput_tok_s": 30.94,
        "status": "Coherent / Fluent Baseline",
    },
    {
        "strategy": "StreamingLLM (Cap=64, Sink=4)",
        "type": "Token Eviction",
        "capacity": 64,
        "sink_tokens": 4,
        "retained_tokens": 64,
        "kv_mb": 0.7500,
        "kv_bytes": 786432,
        "compression_pct": 71.17,
        "compression_ratio": 3.47,
        "latency_sec": 4.827,
        "throughput_tok_s": 31.07,
        "status": "Verified / RoPE Monotonicity Intact",
    },
    {
        "strategy": "StreamingLLM (Cap=40, Sink=4)",
        "type": "Token Eviction",
        "capacity": 40,
        "sink_tokens": 4,
        "retained_tokens": 40,
        "kv_mb": 0.4688,
        "kv_bytes": 491520,
        "compression_pct": 81.98,
        "compression_ratio": 5.55,
        "latency_sec": 4.890,
        "throughput_tok_s": 30.67,
        "status": "Verified / RoPE Monotonicity Intact",
    },
]

DEFAULTS = {
    "seq_len": 512,
    "num_heads": 4,
    "head_dim": 64,
    "chunk_size": 16,
    "sink_tokens": 4,
    "recent_tokens": 32,
    "target_budget": 256,
    "bits": 4,
    "alloc_policy": "importance_weighted",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_hardware_info() -> Dict[str, Any]:
    try:
        info = MemoryMonitor.get_hardware_info()
        if isinstance(info, dict):
            return info
    except Exception:
        pass
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count": __import__("os").cpu_count() or 1,
        "total_system_ram_mb": 0,
        "cuda_available": torch.cuda.is_available(),
        "gpu": None,
    }


def fmt_mb(value: Any, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f} MB"
    except (TypeError, ValueError):
        return "—"


def safe_num(mapping: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(mapping.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def current_config() -> Dict[str, Any]:
    return {key: st.session_state[key] for key in DEFAULTS}


def config_signature(config: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(config[k] for k in DEFAULTS)


def reset_configuration() -> None:
    # Streamlit executes callbacks before the next script run, so it is safe
    # to update widget state here (rather than after widgets are rendered).
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["last_run"] = None
    st.session_state["last_run_signature"] = None
    st.session_state["run_error"] = None
    st.session_state["reset_notice"] = True


def navigate_to(page: str) -> None:
    # Navigation changes are applied through a callback before the radio
    # widget is instantiated on the next Streamlit rerun.
    st.session_state["nav_page"] = page


def html_kpi(label: str, value: str, note: str = "", tone: str = "default") -> None:
    tone_class = {"green": "color:#16834A", "purple": "color:#5B3CE6", "blue": "color:#2875D4"}.get(tone, "")
    st.markdown(
        f"""<div class="fc-kpi">
          <div class="fc-kpi-label">{html.escape(label)}</div>
          <div class="fc-kpi-value" style="{tone_class}">{html.escape(value)}</div>
          <div class="fc-kpi-note">{html.escape(note)}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, page_tag: Optional[str] = None) -> None:
    """Render a responsive page heading with badges kept in the content flow."""
    heading_col, badges_col = st.columns([4.4, 1.6], gap="medium")
    with heading_col:
        if page_tag:
            st.markdown(
                f'<div class="fc-small-label" style="color:{PURPLE};margin-bottom:5px">{html.escape(page_tag)}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="fc-page-title">{html.escape(title)}</div>'
            f'<div class="fc-page-sub">{html.escape(subtitle)}</div>',
            unsafe_allow_html=True,
        )
    with badges_col:
        backend = "CUDA" if torch.cuda.is_available() else "CPU"
        badge_html = (
            '<div class="fc-header-badge-wrap"><div class="fc-badges">'
            '<span class="fc-badge fc-badge-green"><span class="fc-dot"></span> Local demo</span>'
            f'<span class="fc-badge">▦ &nbsp; {backend} backend</span>'
            '</div></div>'
        )
        st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown('<div class="fc-header-rule"></div>', unsafe_allow_html=True)


def plotly_style(fig: go.Figure, height: int = 330, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=26, b=18),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, Arial, sans-serif", color=NAVY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        showlegend=show_legend,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER, font_color=NAVY),
    )
    fig.update_xaxes(showgrid=False, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor="#EDF0F5", zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = torch.randn(1, config["num_heads"], config["seq_len"], config["head_dim"], dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    runtime = None
    start = time.perf_counter()
    try:
        runtime = FlashCacheRuntime(
            chunk_size=config["chunk_size"],
            sink_tokens=config["sink_tokens"],
            recent_tokens=config["recent_tokens"],
            total_budget=config["target_budget"],
            bits=config["bits"],
            allocation_policy=config["alloc_policy"],
            device=device,
        )
        ranking = runtime.compact_layer(keys=keys, values=values, layer_idx=0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stats = runtime.get_memory_stats()
        past_k, past_v = runtime.get_past_key_values(layer_idx=0, target_dtype=torch.bfloat16)

        protected_tokens = int(ranking.total_protected_tokens)
        selected_tokens = int(ranking.total_selected_tokens)
        evicted_tokens = int(ranking.total_evicted_tokens)
        input_tokens = int(config["num_heads"] * config["seq_len"])
        total_chunks = len(ranking.protected_chunks) + len(ranking.selected_chunks) + len(ranking.evicted_chunks)
        quantized_chunks = [c for c in ranking.selected_chunks if c.compressed_key is not None and c.compressed_value is not None]
        failed_chunks = len(ranking.selected_chunks) - len(quantized_chunks)

        key_metrics: List[Dict[str, float]] = []
        value_metrics: List[Dict[str, float]] = []
        for chunk in quantized_chunks:
            metrics = chunk.metadata.extra.get("reconstruction_metrics", {})
            if metrics.get("key"):
                key_metrics.append(metrics["key"])
            if metrics.get("value"):
                value_metrics.append(metrics["value"])

        def avg_metric(rows: List[Dict[str, float]], name: str) -> Optional[float]:
            vals = [float(row[name]) for row in rows if name in row and math.isfinite(float(row[name]))]
            return sum(vals) / len(vals) if vals else None

        shape_k = list(past_k.shape) if past_k is not None else None
        shape_v = list(past_v.shape) if past_v is not None else None
        input_bytes = int(keys.numel() * keys.element_size() * 2)
        original_bytes = int(safe_num(stats, "original_kv_bytes", input_bytes))
        compacted_bytes = int(safe_num(stats, "compacted_kv_bytes", 0))

        return {
            "config": dict(config),
            "device": device,
            "elapsed_ms": float(elapsed_ms),
            "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))},
            "input_tokens": input_tokens,
            "total_chunks": total_chunks,
            "protected_chunks": len(ranking.protected_chunks),
            "selected_chunks": len(ranking.selected_chunks),
            "evicted_chunks": len(ranking.evicted_chunks),
            "protected_tokens": protected_tokens,
            "selected_tokens": selected_tokens,
            "evicted_tokens": evicted_tokens,
            "quantized_chunks": len(quantized_chunks),
            "failed_chunks": failed_chunks,
            "original_bytes": original_bytes,
            "compacted_bytes": compacted_bytes,
            "original_mb": safe_num(stats, "original_kv_mb", original_bytes / (1024 ** 2)),
            "compacted_mb": safe_num(stats, "compacted_kv_mb", compacted_bytes / (1024 ** 2)),
            "compression_ratio": safe_num(stats, "compression_ratio", 0.0),
            "memory_reduction_pct": safe_num(stats, "memory_reduction_pct", 0.0),
            "key_shape": shape_k,
            "value_shape": shape_v,
            "key_psnr": avg_metric(key_metrics, "psnr"),
            "value_psnr": avg_metric(value_metrics, "psnr"),
            "key_cosine": avg_metric(key_metrics, "cosine_similarity"),
            "value_cosine": avg_metric(value_metrics, "cosine_similarity"),
        }
    finally:
        if runtime is not None:
            runtime.shutdown()


@st.cache_data(show_spinner=False)
def quantization_analysis(chunk_size: int, head_dim: int) -> Dict[str, Any]:
    """Run an actual, reproducible CPU-side INT4/INT2 key and value analysis."""
    torch.manual_seed(2026)
    key = torch.randn(1, 1, chunk_size, head_dim, dtype=torch.bfloat16, device="cpu")
    value = torch.randn(1, 1, chunk_size, head_dim, dtype=torch.bfloat16, device="cpu")
    quantizer4, quantizer2 = KIVIQuantizer(bits=4), KIVIQuantizer(bits=2)
    dequantizer = KIVIDequantizer()
    output: Dict[str, Any] = {"metrics": {}, "errors": {"key": {}, "value": {}}}
    for bits, quantizer in ((4, quantizer4), (2, quantizer2)):
        for label, tensor, method in (("key", key, quantizer.quantize_key), ("value", value, quantizer.quantize_value)):
            compressed = method(tensor)
            reconstructed = dequantizer.dequantize(compressed, target_dtype=torch.bfloat16)
            metrics = dequantizer.evaluate_reconstruction(tensor, reconstructed)
            original_bytes = tensor.numel() * tensor.element_size()
            metrics = {k: float(v) for k, v in metrics.items()}
            metrics["representation_ratio"] = float(original_bytes / max(1, compressed.total_bytes))
            metrics["compressed_bytes"] = int(compressed.total_bytes)
            output["metrics"][f"{label}_{bits}"] = metrics
            output["errors"][label][str(bits)] = (tensor.float() - reconstructed.float()).abs().flatten().tolist()
    return output


def token_counts(run: Dict[str, Any]) -> Dict[str, int]:
    return {
        "Protected": int(run["protected_tokens"]),
        "Selected": int(run["selected_tokens"]),
        "Evicted": int(run["evicted_tokens"]),
    }


def render_metric_row(run: Dict[str, Any]) -> None:
    cols = st.columns(4, gap="medium")
    with cols[0]:
        html_kpi("Original KV memory", fmt_mb(run["original_mb"]), "BF16 input · keys + values")
    with cols[1]:
        html_kpi("Compacted KV memory", fmt_mb(run["compacted_mb"]), "Retained cache representation", "green")
    with cols[2]:
        ratio = run["compression_ratio"]
        html_kpi("End-to-end compaction ratio", f"{ratio:.2f}×" if ratio else "—", "Eviction + quantization", "blue")
    with cols[3]:
        html_kpi("Total memory reduction", f"{run['memory_reduction_pct']:.2f}%", "Relative to original KV", "green")


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
st.session_state.setdefault("last_run", None)
st.session_state.setdefault("last_run_signature", None)
st.session_state.setdefault("run_error", None)
st.session_state.setdefault("reset_notice", False)
st.session_state.setdefault("nav_page", "Live Demo")
st.session_state.setdefault("quant_chunk", 16)

hw = get_hardware_info()

# -----------------------------------------------------------------------------
# Sidebar: navigation and machine status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """<div class="fc-brand"><div class="fc-logo">◇</div><div>
        <div class="fc-brand-title">FlashCache-KV</div><div class="fc-brand-sub">KV CACHE OPTIMIZER</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fc-side-label">WORKSPACE</div>', unsafe_allow_html=True)
    nav_options = ["Dashboard", "Live Demo", "Memory Analytics", "Quantization Analysis", "Historical Benchmarks", "Settings"]
    nav_icons = {
        "Dashboard": "⌂  Dashboard",
        "Live Demo": "▷  Live Demo",
        "Memory Analytics": "▥  Memory Analytics",
        "Quantization Analysis": "◈  Quantization Analysis",
        "Historical Benchmarks": "♜  Historical Benchmarks",
        "Settings": "⚙  Settings",
    }
    current_index = nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 1
    selected_page = st.radio(
        "Workspace", nav_options, index=current_index, key="nav_page",
        format_func=lambda page: nav_icons[page], label_visibility="collapsed",
    )

    device_label = str(hw.get("device", "cuda" if torch.cuda.is_available() else "cpu")).upper()
    ram_gb = safe_num(hw, "total_system_ram_mb", 0.0) / 1024
    ram_text = f"{ram_gb:.1f} GB" if ram_gb else "Unavailable"
    system_rows = [
        ("Device", device_label),
        ("CPU threads", str(int(torch.get_num_threads()))),
        ("System RAM", ram_text),
        ("PyTorch", torch.__version__),
    ]
    if hw.get("cuda_available") and hw.get("gpu"):
        system_rows.append(("GPU", str(hw["gpu"].get("device_name", "NVIDIA GPU"))))
    rows_html = "".join(
        f'<div class="fc-system-row"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in system_rows
    )
    st.markdown(
        f'<div class="fc-system"><div class="fc-system-title">▣ &nbsp; System</div>{rows_html}'
        '<div class="fc-ready"><span class="fc-dot"></span> Ready</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.caption("Local research prototype")


# -----------------------------------------------------------------------------
# Dashboard overview
# -----------------------------------------------------------------------------
if selected_page == "Dashboard":
    page_header("Dashboard", "A focused overview of FlashCache-KV execution, memory behaviour and reconstruction quality.")
    st.markdown('<div class="fc-section-title">Workspace overview</div><div class="fc-section-sub">Choose a workspace to inspect or run the pipeline.</div>', unsafe_allow_html=True)
    nav_cards = [
        ("Live Demo", "Configure a synthetic BF16 workload and execute the five-stage pipeline."),
        ("Memory Analytics", "Inspect cache footprint, memory savings and context-length scaling."),
        ("Quantization Analysis", "Compare INT4 and INT2 reconstruction fidelity for keys and values."),
        ("Historical Benchmarks", "Review historical reference measurements stored in the project."),
    ]
    c1, c2 = st.columns(2, gap="large")
    for i, (title, description) in enumerate(nav_cards):
        with (c1 if i % 2 == 0 else c2):
            with st.container(border=True):
                st.markdown(f"<div class='fc-section-title'>{html.escape(title)}</div><div class='fc-section-sub'>{html.escape(description)}</div>", unsafe_allow_html=True)
                st.button(
                    f"Open {title}", key=f"open_{i}", use_container_width=True,
                    on_click=navigate_to, args=(title,),
                )
    st.markdown("### Latest local execution")
    if st.session_state["last_run"]:
        run = st.session_state["last_run"]
        render_metric_row(run)
        st.caption(f"Last run: {run['elapsed_ms']:.2f} ms · {run['device'].upper()} · INT{run['config']['bits']}")
    else:
        st.markdown('<div class="fc-empty">No local run has been recorded in this session yet.<br>Open <b>Live Demo</b> to execute FlashCache-KV.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Live demo
# -----------------------------------------------------------------------------
elif selected_page == "Live Demo":
    page_header("Live Demo", "Configure a synthetic BF16 KV workload and execute the five-stage FlashCache-KV compaction pipeline.", "LIVE EXECUTION")

    with st.container(border=True):
        head_cols = st.columns([4, 1.4])
        with head_cols[0]:
            st.markdown('<div class="fc-section-title">Run configuration</div><div class="fc-section-sub">Set the input workload, cache policy and compression strategy.</div>', unsafe_allow_html=True)
        with head_cols[1]:
            st.button("↺  Reset configuration", key="reset_live_config", on_click=reset_configuration, use_container_width=True)

        c_work, c_cache, c_compress = st.columns([1.08, 1.14, 1.0], gap="medium")
        with c_work:
            with st.container(border=True):
                st.markdown("#### Workload")
                st.caption("Configure the input KV workload.")
                st.slider("Context length", 128, 4096, step=64, key="seq_len", help="Number of tokens in each input sequence.")
                a, b = st.columns(2, gap="small")
                with a:
                    st.slider("KV attention heads", 1, 16, step=1, key="num_heads")
                with b:
                    st.selectbox("Head dimension", [32, 64, 128], key="head_dim")
                input_mb = (2 * st.session_state.num_heads * st.session_state.seq_len * st.session_state.head_dim * 2) / (1024 ** 2)
                st.markdown(f'<div class="fc-note">Input shape: <b>[1, {st.session_state.num_heads}, {st.session_state.seq_len:,}, {st.session_state.head_dim}]</b> · BF16 · estimated input {input_mb:.3f} MB</div>', unsafe_allow_html=True)

        with c_cache:
            with st.container(border=True):
                st.markdown("#### Cache policy")
                st.caption("Chunking, retention and budget allocation.")
                a, b = st.columns(2, gap="small")
                with a:
                    st.selectbox("Chunk size", [8, 16, 32], key="chunk_size")
                    st.selectbox("Recent local window", [8, 16, 24, 32, 40, 48, 56, 64], key="recent_tokens")
                with b:
                    st.selectbox("Attention sinks", [0, 2, 4, 6, 8, 10, 12, 14, 16], key="sink_tokens")
                    st.slider("Target retention budget", 64, 1024, step=32, key="target_budget")
                protected_floor = min(st.session_state.seq_len, st.session_state.sink_tokens + st.session_state.recent_tokens) * st.session_state.num_heads
                available = max(0, st.session_state.target_budget - protected_floor)
                st.markdown(f'<div class="fc-note">Protected-token floor estimate: <b>{protected_floor:,}</b> · Budget remaining for selected chunks: <b>{available:,}</b></div>', unsafe_allow_html=True)

        with c_compress:
            with st.container(border=True):
                st.markdown("#### Compression")
                st.caption("Quantization and allocation strategy.")
                st.selectbox("Quantization precision", [4, 2], format_func=lambda x: f"INT{x} ({x}-bit)", key="bits")
                st.selectbox(
                    "Budget allocation policy",
                    ["importance_weighted", "uniform", "recency_aware"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="alloc_policy",
                )
                st.markdown('<div class="fc-note">Keys use per-channel quantization; values use per-token quantization in the KIVI-inspired path.</div>', unsafe_allow_html=True)

        action_left, action_right = st.columns([1.75, 4.5], gap="medium")
        with action_left:
            run_clicked = st.button("▶  Run FlashCache-KV", type="primary", use_container_width=True, key="run_flashcache")
        with action_right:
            st.markdown('<div style="min-height:48px;display:flex;align-items:center;padding:.35rem .2rem;color:#64718B;font-size:.82rem;line-height:1.45">Executes chunking → adaptive budgeting → ranking → quantization → cache reconstruction.</div>', unsafe_allow_html=True)

    if st.session_state.get("reset_notice"):
        st.success("Configuration restored to defaults. Previous execution results were cleared.")
        st.session_state["reset_notice"] = False

    config = current_config()
    signature = config_signature(config)
    if run_clicked:
        st.session_state["run_error"] = None
        try:
            with st.spinner("Running the FlashCache-KV pipeline on the selected device…"):
                st.session_state["last_run"] = run_pipeline(config)
                st.session_state["last_run_signature"] = signature
        except Exception as exc:
            st.session_state["run_error"] = f"{type(exc).__name__}: {exc}"
            st.session_state["last_run"] = None
            st.session_state["last_run_signature"] = None

    if st.session_state.get("run_error"):
        st.error("Pipeline execution failed. The runtime returned an error; previous results are not being presented as current.")
        with st.expander("Technical error details"):
            st.code(st.session_state["run_error"])

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        title_cols = st.columns([3.6, 2])
        with title_cols[0]:
            st.markdown('<div class="fc-section-title">Execution results</div><div class="fc-section-sub">Measured outputs from the most recently completed local run.</div>', unsafe_allow_html=True)
        run = st.session_state.get("last_run")
        if run:
            with title_cols[1]:
                st.markdown(f'<div style="text-align:right;color:{MUTED};font-size:.73rem;padding-top:7px">Last run: <b>{run["elapsed_ms"]:.2f} ms</b> &nbsp;·&nbsp; Device: <b>{run["device"].upper()}</b> &nbsp;·&nbsp; Precision: <b>INT{run["config"]["bits"]}</b></div>', unsafe_allow_html=True)
            current_match = st.session_state.get("last_run_signature") == signature
            if current_match:
                st.markdown(f'<div class="fc-status"><span class="fc-dot"></span><b>Execution completed successfully</b><span>·</span>{run["elapsed_ms"]:.2f} ms<span>·</span>{run["device"].upper()} backend</div>', unsafe_allow_html=True)
            else:
                st.warning("Configuration has changed since the last run. Results below belong to the previous configuration; run again to refresh them.")
            render_metric_row(run)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            left, right = st.columns([1.25, 1], gap="medium")
            with left:
                with st.container(border=True):
                    st.markdown('<div class="fc-section-title">Five-stage pipeline</div><div class="fc-section-sub">Execution audit from the actual local runtime.</div>', unsafe_allow_html=True)
                    stages = [
                        ("ChunkKV construction", f"Partitioned {run['total_chunks']:,} candidate chunks."),
                        ("Adaptive budget allocation", f"Distributed retention budget across {run['config']['num_heads']} attention heads."),
                        ("Importance-based ranking", f"Selected {run['selected_chunks']:,} chunks · evicted {run['evicted_chunks']:,} chunks."),
                        ("KIVI-inspired quantization", (f"Successfully compressed {run['quantized_chunks']:,} selected chunks using INT{run['config']['bits']}." if run["quantized_chunks"] else "No middle chunks were selected for quantization; protected tokens remain in BF16.")),
                        ("Cache reconstruction", f"Reconstructed keys and values with shape {run['key_shape']}.")
                    ]
                    for idx, (name, description) in enumerate(stages, start=1):
                        row = st.columns([0.42, 3.7, 1.0])
                        with row[0]:
                            st.markdown(f'<div class="fc-stage-num">{idx}</div>', unsafe_allow_html=True)
                        with row[1]:
                            st.markdown(f'<div class="fc-stage-title">{html.escape(name)}</div><div class="fc-stage-desc">{html.escape(description)}</div>', unsafe_allow_html=True)
                        with row[2]:
                            if idx == 4 and run["selected_chunks"] > 0 and run["quantized_chunks"] == 0:
                                st.markdown('<span class="fc-pill fc-muted-pill">Not compressed</span>', unsafe_allow_html=True)
                            elif idx == 4 and run["failed_chunks"] > 0:
                                st.markdown('<span class="fc-pill fc-muted-pill">Partial</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="fc-pill">✓ Completed</span>', unsafe_allow_html=True)
                        if idx < 5:
                            st.markdown('<div style="height:7px;border-bottom:1px solid #EDF0F5;margin-bottom:8px"></div>', unsafe_allow_html=True)
                    if run["failed_chunks"]:
                        st.warning(f"{run['failed_chunks']} selected chunk(s) did not complete compression and retained their original representation.")
                    if run["key_psnr"] is not None:
                        st.caption(f"Mean reconstruction PSNR · Key: {run['key_psnr']:.2f} dB · Value: {run['value_psnr']:.2f} dB")
            with right:
                with st.container(border=True):
                    st.markdown('<div class="fc-section-title">Token disposition</div><div class="fc-section-sub">How the input tokens were handled after ranking.</div>', unsafe_allow_html=True)
                    counts = token_counts(run)
                    labels = list(counts.keys())
                    values = list(counts.values())
                    if sum(values) > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=labels, values=values, hole=.68, sort=False,
                            marker=dict(colors=[PURPLE, BLUE, "#D6DCE7"], line=dict(color="#FFFFFF", width=3)),
                            textinfo="percent", textposition="inside", hovertemplate="%{label}: %{value:,} tokens (%{percent})<extra></extra>",
                        )])
                        fig.update_layout(annotations=[dict(text=f"<b>{sum(values):,}</b><br><span style='font-size:11px'>tokens</span>", x=.5, y=.5, showarrow=False, font=dict(size=19, color=NAVY))])
                        plotly_style(fig, height=250, show_legend=False)
                        fig.update_layout(margin=dict(l=5, r=5, t=4, b=4))
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    for label, color in zip(labels, [PURPLE, BLUE, "#AAB4C4"]):
                        count = counts[label]
                        pct = (count / max(1, run["input_tokens"])) * 100
                        st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 0;font-size:.76rem"><span><span style="display:inline-block;width:9px;height:9px;border-radius:3px;background:{color};margin-right:7px"></span>{label}</span><b>{count:,} &nbsp; <span style="color:{MUTED};font-weight:500">{pct:.1f}%</span></b></div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<div class="fc-section-title">Reconstruction details</div><div class="fc-section-sub">Tensor shapes and runtime information from the latest execution.</div>', unsafe_allow_html=True)
                d1, d2, d3, d4 = st.columns(4, gap="medium")
                with d1:
                    st.markdown('<div class="fc-small-label">Key tensor shape</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="fc-detail-value">{html.escape(str(run["key_shape"]))}</div>', unsafe_allow_html=True)
                with d2:
                    st.markdown('<div class="fc-small-label">Value tensor shape</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="fc-detail-value">{html.escape(str(run["value_shape"]))}</div>', unsafe_allow_html=True)
                with d3:
                    st.markdown('<div class="fc-small-label">Input dtype</div><div class="fc-detail-value">BF16</div>', unsafe_allow_html=True)
                with d4:
                    st.markdown('<div class="fc-small-label">Execution device</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="fc-detail-value">{run["device"].upper()}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="fc-empty">Configure a workload above and select <b>Run FlashCache-KV</b>.<br>Your actual runtime metrics, stage audit and reconstructed tensor shapes will appear here.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Memory analytics
# -----------------------------------------------------------------------------
elif selected_page == "Memory Analytics":
    page_header("Memory Analytics", "Analyze KV cache memory usage, savings and scaling across different retention and compression strategies.")
    run = st.session_state.get("last_run")
    if run:
        render_metric_row(run)
        stats = run["stats"]
        st.markdown("### KV cache memory comparison")
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            with st.container(border=True):
                df_memory = pd.DataFrame({"Representation": ["Original BF16 input", "Compacted retained cache"], "Memory (MB)": [run["original_mb"], run["compacted_mb"]]})
                fig = px.bar(df_memory, x="Memory (MB)", y="Representation", orientation="h", text="Memory (MB)", color="Representation", color_discrete_map={"Original BF16 input": "#AEB8C8", "Compacted retained cache": GREEN})
                fig.update_traces(texttemplate="%{text:.3f} MB", textposition="outside", cliponaxis=False)
                fig.update_layout(showlegend=False, xaxis_title="Memory (MB)", yaxis_title="", xaxis=dict(rangemode="tozero"))
                plotly_style(fig, height=270, show_legend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with right:
            with st.container(border=True):
                st.markdown("#### Physical block state distribution")
                state_values = [
                    max(0, int(safe_num(stats, "active_bf16_blocks"))),
                    max(0, int(safe_num(stats, "compressed_blocks"))),
                    max(0, int(safe_num(stats, "reclaimable_blocks"))),
                ]
                state_labels = ["Active BF16", "Compressed", "Reclaimable source"]
                if sum(state_values) > 0:
                    fig = go.Figure(go.Pie(labels=state_labels, values=state_values, hole=.65, sort=False,
                        marker=dict(colors=[BLUE, PURPLE, GREEN], line=dict(color="#FFFFFF", width=3)),
                        hovertemplate="%{label}: %{value} blocks (%{percent})<extra></extra>"))
                    fig.update_layout(annotations=[dict(text=f"<b>{sum(state_values)}</b><br><span style='font-size:10px'>blocks</span>", x=.5, y=.5, showarrow=False, font=dict(size=18, color=NAVY))])
                    plotly_style(fig, height=260, show_legend=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("The runtime did not return block-state counts for this run.")
                st.caption("Reclaimable is an allocator lifecycle state; it is not a direct measurement of operating-system RAM returned.")

        st.markdown("### Where the memory reduction comes from")
        total_tokens = max(1, run["input_tokens"])
        retained_tokens = run["protected_tokens"] + run["selected_tokens"]
        retained_bf16_bytes = run["original_bytes"] * (retained_tokens / total_tokens)
        eviction_bytes = max(0.0, run["original_bytes"] - retained_bf16_bytes)
        quantization_delta_bytes = retained_bf16_bytes - run["compacted_bytes"]
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            html_kpi("Estimated reduction from eviction", f"{(eviction_bytes / max(1, run['original_bytes']))*100:.1f}%", f"{eviction_bytes / (1024**2):.3f} MB estimated", "blue")
        with c2:
            if quantization_delta_bytes >= 0:
                html_kpi("Additional reduction after retention", f"{(quantization_delta_bytes / max(1, retained_bf16_bytes))*100:.1f}%", f"{quantization_delta_bytes / (1024**2):.3f} MB estimated", "green")
            else:
                html_kpi("Additional storage overhead", f"{abs(quantization_delta_bytes) / max(1, retained_bf16_bytes)*100:.1f}%", f"{abs(quantization_delta_bytes) / (1024**2):.3f} MB above retained BF16 estimate", "default")
        with c3:
            html_kpi("Total end-to-end reduction", f"{run['memory_reduction_pct']:.1f}%", f"{run['original_mb']:.3f} → {run['compacted_mb']:.3f} MB", "green")
        st.caption("Eviction and post-retention deltas are estimates based on an equal per-token BF16 footprint. The end-to-end total comes from the runtime's memory accounting.")
    else:
        st.markdown('<div class="fc-empty">Run the Live Demo first to populate measured memory analytics. The scaling chart below is a modelled estimate from the prototype formula, not a hardware benchmark.</div>', unsafe_allow_html=True)

    st.markdown("### Comparative memory scaling")
    st.caption("Modelled estimates using the original dashboard assumptions. They are not measured FlashCache-KV execution results.")
    cfg = current_config()
    test_lengths = [512, 1024, 2048, 4096]
    comp_data = []
    for length in test_lengths:
        full_mb = (2 * cfg["num_heads"] * length * cfg["head_dim"] * 2) / (1024 ** 2)
        evict_mb = (2 * cfg["num_heads"] * min(length, cfg["target_budget"]) * cfg["head_dim"] * 2) / (1024 ** 2)
        for strategy, value in [
            ("Full BF16 cache", full_mb),
            ("StreamingLLM eviction", evict_mb),
            ("ChunkKV uncompressed", evict_mb),
            ("FlashCache-KV INT4 (estimate)", evict_mb * 0.28),
            ("FlashCache-KV INT2 (estimate)", evict_mb * 0.18),
        ]:
            comp_data.append({"Context length": str(length), "Strategy": strategy, "KV memory (MB)": value})
    df_scale = pd.DataFrame(comp_data)
    with st.container(border=True):
        fig = px.bar(df_scale, x="Context length", y="KV memory (MB)", color="Strategy", barmode="group",
            color_discrete_sequence=["#AEB8C8", "#E5B84B", "#5B9BE8", "#24A36A", "#9A6AE8"])
        fig.update_layout(xaxis_title="Context length (tokens)", yaxis_title="KV memory (MB)", legend_title="")
        plotly_style(fig, height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# -----------------------------------------------------------------------------
# Quantization analysis
# -----------------------------------------------------------------------------
elif selected_page == "Quantization Analysis":
    page_header("Quantization Analysis", "Measure reconstruction fidelity and storage trade-offs for KIVI-inspired INT4 and INT2 key/value quantization.")
    controls = st.columns([1.2, 1.2, 3])
    with controls[0]:
        q_chunk = st.selectbox("Analysis chunk size", [8, 16, 32, 64], index=1, key="quant_chunk")
    with controls[1]:
        st.selectbox("Analysis head dimension", [32, 64, 128], index=[32, 64, 128].index(st.session_state.head_dim) if st.session_state.head_dim in [32, 64, 128] else 1, key="quant_head_dim")
    with controls[2]:
        st.markdown('<div style="padding-top:1.8rem;color:#64718B;font-size:.78rem">Reproducible synthetic BF16 tensors · analysis runs on CPU for consistent comparison.</div>', unsafe_allow_html=True)

    q_results = quantization_analysis(int(q_chunk), int(st.session_state["quant_head_dim"]))
    m = q_results["metrics"]
    k4, v4, k2, v2 = m["key_4"], m["value_4"], m["key_2"], m["value_2"]

    top = st.columns(4, gap="medium")
    with top[0]: html_kpi("INT4 · Key PSNR", f"{k4['psnr']:.2f} dB", "Higher is better", "purple")
    with top[1]: html_kpi("INT2 · Key PSNR", f"{k2['psnr']:.2f} dB", "Higher is better")
    with top[2]: html_kpi("INT4 · Key representation", f"{k4['representation_ratio']:.2f}×", "Original tensor bytes / packed bytes", "green")
    with top[3]: html_kpi("INT2 · Key representation", f"{k2['representation_ratio']:.2f}×", "Includes scales and zero points", "blue")

    st.markdown("### Quantization fidelity")
    metric_rows = []
    for label, kmetric, vmetric in [("PSNR (dB) ↑", k4, k2), ("Cosine similarity ↑", k4, k2), ("Normalized MSE ↓", k4, k2), ("Representation ratio (key)", k4, k2)]:
        metric_key = {"PSNR (dB) ↑": "psnr", "Cosine similarity ↑": "cosine_similarity", "Normalized MSE ↓": "nmse", "Representation ratio (key)": "representation_ratio"}[label]
        metric_rows.append({"Metric": label, "INT4 · Key": k4[metric_key], "INT4 · Value": v4[metric_key], "INT2 · Key": k2[metric_key], "INT2 · Value": v2[metric_key]})
    df_metrics = pd.DataFrame(metric_rows)
    for col in df_metrics.columns[1:]:
        df_metrics[col] = df_metrics[col].map(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)
    with st.container(border=True):
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    st.markdown("### Reconstruction error distribution")
    e1, e2 = st.columns(2, gap="large")
    for col, tensor_name, title in [(e1, "key", "Key reconstruction error"), (e2, "value", "Value reconstruction error")]:
        with col:
            with st.container(border=True):
                st.markdown(f"#### {title}")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=q_results["errors"][tensor_name]["4"], name="INT4", opacity=.68, nbinsx=36, marker_color=PURPLE, histnorm="probability density"))
                fig.add_trace(go.Histogram(x=q_results["errors"][tensor_name]["2"], name="INT2", opacity=.56, nbinsx=36, marker_color=BLUE, histnorm="probability density"))
                fig.update_layout(barmode="overlay", xaxis_title="Absolute reconstruction error", yaxis_title="Probability density")
                plotly_style(fig, height=320)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Key and value quality by precision")
    scatter_rows = []
    for precision, key_metric, value_metric in [("INT4", k4, v4), ("INT2", k2, v2)]:
        scatter_rows.append({"Tensor": f"{precision} · Key", "Representation ratio": key_metric["representation_ratio"], "PSNR (dB)": key_metric["psnr"], "Precision": precision})
        scatter_rows.append({"Tensor": f"{precision} · Value", "Representation ratio": value_metric["representation_ratio"], "PSNR (dB)": value_metric["psnr"], "Precision": precision})
    df_scatter = pd.DataFrame(scatter_rows)
    with st.container(border=True):
        fig = px.scatter(df_scatter, x="Representation ratio", y="PSNR (dB)", color="Precision", symbol="Tensor", text="Tensor",
            color_discrete_map={"INT4": PURPLE, "INT2": BLUE})
        fig.update_traces(textposition="top center", marker=dict(size=12))
        fig.update_layout(xaxis_title="Representation ratio (×; includes metadata)", yaxis_title="Reconstruction PSNR (dB)", legend_title="")
        plotly_style(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Analysis uses synthetic BF16 tensors with shape [1, 1, {q_chunk}, {st.session_state['quant_head_dim']}]. Key quantization is per-channel; value quantization is per-token. Results are measured by the project's KIVIDequantizer.")


# -----------------------------------------------------------------------------
# Historical benchmarks
# -----------------------------------------------------------------------------
elif selected_page == "Historical Benchmarks":
    page_header("Historical Benchmarks", "Reference records from the original FlashCache-KV dashboard source. These are not measurements from the current local run.")
    st.info("Historical reference data · Attlornets / Qwen2.5-0.5B-Instruct baseline context as identified in the original project source.")
    df_hist = pd.DataFrame(HISTORICAL_BENCHMARKS)
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1: html_kpi("Reference strategies", f"{len(df_hist)}", "Stored records")
    with k2: html_kpi("Lowest reference KV", f"{df_hist['kv_mb'].min():.4f} MB", "Among listed records", "green")
    with k3: html_kpi("Highest listed ratio", f"{df_hist['compression_ratio'].max():.2f}×", "Stored compression ratio", "blue")
    with k4: html_kpi("Latency range", f"{df_hist['latency_sec'].min():.3f}–{df_hist['latency_sec'].max():.3f}s", "Historical records")

    st.markdown("### Historical memory footprint")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True):
            fig = px.bar(df_hist, x="kv_mb", y="strategy", orientation="h", text="kv_mb", color="type",
                color_discrete_map={"Full Precision": "#AEB8C8", "Token Eviction": BLUE})
            fig.update_traces(texttemplate="%{text:.4f} MB", textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title="KV memory (MB)", yaxis_title="", legend_title="Strategy type")
            plotly_style(fig, height=310)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_hist["strategy"], y=df_hist["latency_sec"], marker_color=["#AEB8C8", BLUE, BLUE], name="Latency"))
            fig.update_layout(xaxis_title="Reference strategy", yaxis_title="Latency (seconds)")
            fig.update_xaxes(tickangle=-16)
            plotly_style(fig, height=310, show_legend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Reference results")
    display_cols = ["strategy", "type", "capacity", "sink_tokens", "retained_tokens", "kv_mb", "compression_ratio", "compression_pct", "latency_sec", "throughput_tok_s", "status"]
    display_df = df_hist[display_cols].rename(columns={
        "strategy": "Strategy", "type": "Method type", "capacity": "Capacity", "sink_tokens": "Sink tokens",
        "retained_tokens": "Retained tokens", "kv_mb": "KV memory (MB)", "compression_ratio": "Compression ratio (×)",
        "compression_pct": "Memory reduction (%)", "latency_sec": "Latency (s)", "throughput_tok_s": "Throughput (tokens/s)", "status": "Reference status",
    })
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.caption("The values above are the three records embedded in the original app.py. They are preserved as supplied; no additional benchmark rows or results have been invented.")


# -----------------------------------------------------------------------------
# Settings / environment
# -----------------------------------------------------------------------------
elif selected_page == "Settings":
    page_header("Settings", "Environment details and current dashboard defaults.")
    st.markdown("### Runtime environment")
    env_rows = [
        {"Property": "Python", "Value": platform.python_version()},
        {"Property": "PyTorch", "Value": torch.__version__},
        {"Property": "Operating system", "Value": platform.platform()},
        {"Property": "Selected automatic backend", "Value": "CUDA" if torch.cuda.is_available() else "CPU"},
        {"Property": "CUDA available", "Value": str(torch.cuda.is_available())},
        {"Property": "CPU threads", "Value": str(torch.get_num_threads())},
        {"Property": "System RAM", "Value": f"{safe_num(hw, 'total_system_ram_mb')/1024:.1f} GB" if safe_num(hw, 'total_system_ram_mb') else "Unavailable"},
    ]
    with st.container(border=True):
        st.dataframe(pd.DataFrame(env_rows), hide_index=True, use_container_width=True)
    st.markdown("### Dashboard defaults")
    defaults_df = pd.DataFrame([{"Parameter": key.replace("_", " ").title(), "Default": str(value)} for key, value in DEFAULTS.items()])
    with st.container(border=True):
        st.dataframe(defaults_df, hide_index=True, use_container_width=True)
    st.caption("This page reports environment and presentation defaults. It does not install packages or alter the FlashCache runtime configuration.")
