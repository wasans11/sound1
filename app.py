import numpy as np
import librosa
import parselmouth
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_frame_jitter_shimmer(path: str, frame_duration: float = 0.025, hop: float = 0.010):
    """
    프레임별 jitter / shimmer 추출.
    - frame_duration: 분석 윈도우 크기 (초)
    - hop: 프레임 간격 (초)
    반환: times(중심 시간), jitters, shimmers (모두 ndarray)
    """
    sound = parselmouth.Sound(path)
    total_dur = sound.duration

    times, jitters, shimmers = [], [], []
    t = frame_duration / 2
    while t < total_dur - frame_duration / 2:
        try:
            chunk = sound.extract_part(
                from_time=t - frame_duration / 2,
                to_time=t + frame_duration / 2,
                preserve_times=False
            )
            pp = parselmouth.praat.call(chunk, "To PointProcess (periodic, cc)", 75, 500)
            j = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            s = parselmouth.praat.call([chunk, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            jitters.append(j * 100)
            shimmers.append(s * 100)
        except Exception:
            jitters.append(0.0)
            shimmers.append(0.0)
        times.append(t)
        t += hop

    return np.array(times), np.array(jitters), np.array(shimmers)


def _normalize(arr):
    """0~1 정규화 (시각화용)"""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def plot_waveform_analysis(
    y_ref: np.ndarray,
    y_input: np.ndarray,
    input_path: str,
    timeline: list,          # get_word_timeline() 결과
    sr: int = 16000,
    word: str = ""
) -> go.Figure:
    """
    파형 + CTC 세그먼트 + 프레임별 Jitter/Shimmer 통합 시각화.

    Parameters
    ----------
    y_ref       : 기준(원어민) 오디오 numpy array
    y_input     : 사용자 오디오 numpy array
    input_path  : 사용자 오디오 파일 경로 (parselmouth용)
    timeline    : [{"Word": str, "Start": float, "End": float}, ...]
    sr          : 샘플링 레이트
    word        : 분석 단어 (제목 표시용)
    """

    # ── 시간축 ──────────────────────────────────────────────
    t_ref   = np.linspace(0, len(y_ref)   / sr, len(y_ref))
    t_input = np.linspace(0, len(y_input) / sr, len(y_input))

    # ── 프레임별 Jitter / Shimmer ───────────────────────────
    js_times, jitters, shimmers = get_frame_jitter_shimmer(input_path)
    jitter_norm  = _normalize(jitters)
    shimmer_norm = _normalize(shimmers)

    # ── 서브플롯 레이아웃 ───────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.06,
        subplot_titles=(
            f"🎵 파형 비교  |  CTC 세그먼트  —  '{word}'",
            "📳 Jitter (목소리 떨림, 프레임별)",
            "🔊 Shimmer (음량 불안정, 프레임별)"
        )
    )

    # ── Row 1 : 파형 ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_ref, y=y_ref,
        mode="lines", name="Native",
        line=dict(color="rgba(100,149,237,0.6)", width=1),
        fill="tozeroy", fillcolor="rgba(100,149,237,0.15)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_input, y=y_input,
        mode="lines", name="Input",
        line=dict(color="rgba(255,99,71,0.7)", width=1),
        fill="tozeroy", fillcolor="rgba(255,99,71,0.15)"
    ), row=1, col=1)

    # ── Row 1 : CTC 세그먼트 박스 + 레이블 ──────────────────
    y_max = max(np.abs(y_ref).max(), np.abs(y_input).max()) * 1.1
    seg_colors = [
        "rgba(255,215,0,0.15)", "rgba(144,238,144,0.15)",
        "rgba(173,216,230,0.15)", "rgba(255,160,122,0.15)",
        "rgba(221,160,221,0.15)"
    ]

    for i, seg in enumerate(timeline):
        color = seg_colors[i % len(seg_colors)]
        # 구간 배경
        fig.add_vrect(
            x0=seg["Start"], x1=seg["End"],
            fillcolor=color, opacity=0.6,
            line_width=1, line_color="rgba(255,255,255,0.3)",
            row=1, col=1
        )
        # 레이블
        fig.add_annotation(
            x=(seg["Start"] + seg["End"]) / 2,
            y=y_max * 0.85,
            text=f"<b>{seg['Word']}</b><br>"
                 f"<span style='font-size:10px'>{seg['End']-seg['Start']:.2f}s</span>",
            showarrow=False,
            font=dict(size=11, color="white"),
            bgcolor="rgba(0,0,0,0.45)",
            bordercolor="white", borderwidth=1,
            row=1, col=1
        )

    # ── Row 2 : Jitter ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=js_times, y=jitters,
        mode="lines", name="Jitter (%)",
        line=dict(color="rgba(255,165,0,0.9)", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,165,0,0.2)"
    ), row=2, col=1)

    # 임계선 (정상 jitter ~1%)
    fig.add_hline(y=1.0, line_dash="dash",
                  line_color="rgba(255,255,255,0.4)",
                  annotation_text="정상 상한(1%)",
                  annotation_position="top right",
                  row=2, col=1)

    # CTC 구간 세로선
    for seg in timeline:
        for t in [seg["Start"], seg["End"]]:
            fig.add_vline(x=t, line_dash="dot",
                          line_color="rgba(255,255,255,0.2)",
                          row=2, col=1)

    # ── Row 3 : Shimmer ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=js_times, y=shimmers,
        mode="lines", name="Shimmer (%)",
        line=dict(color="rgba(0,206,209,0.9)", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,206,209,0.2)"
    ), row=3, col=1)

    # 임계선 (정상 shimmer ~3%)
    fig.add_hline(y=3.0, line_dash="dash",
                  line_color="rgba(255,255,255,0.4)",
                  annotation_text="정상 상한(3%)",
                  annotation_position="top right",
                  row=3, col=1)

    for seg in timeline:
        for t in [seg["Start"], seg["End"]]:
            fig.add_vline(x=t, line_dash="dot",
                          line_color="rgba(255,255,255,0.2)",
                          row=3, col=1)

    # ── 레이아웃 ────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="시간 (초)", row=3, col=1)
    fig.update_yaxes(title_text="진폭", row=1, col=1)
    fig.update_yaxes(title_text="Jitter (%)", row=2, col=1)
    fig.update_yaxes(title_text="Shimmer (%)", row=3, col=1)

    return fig
