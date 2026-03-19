import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial.distance import jensenshannon
from dtaidistance import dtw
from gtts import gTTS
import whisper
import os
import tempfile
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
import parselmouth
from groq import Groq
import soundfile as sf

# --- 0. 설정 ---
st.set_page_config(layout="wide", page_title="말모리 AI 발음 분석기")

WORDS = ["orange", "apple", "banana", "computer", "internet",
         "chocolate", "camera", "energy", "coffee", "television"]

# --- 1. 리소스 로드 ---

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_ctc_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

@st.cache_resource
def load_embedding_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

# --- 2. 오디오 처리 함수 ---

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return librosa.util.normalize(y)

def get_mfcc(y, sr=16000):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)

def get_pitch(y, sr=16000):
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    vals = pitch[pitch > 0]
    return vals.mean() if vals.size > 0 else 0

def get_embedding(y, processor, model):
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def generate_reference(words):
    os.makedirs("reference", exist_ok=True)
    for word in words:
        path = f"reference/{word}.mp3"
        if not os.path.exists(path):
            gTTS(word, lang="en", tld="com").save(path)

def compare(y_ref, y_input, emb_processor, emb_model):
    mfcc_ref = get_mfcc(y_ref)
    mfcc_input = get_mfcc(y_input)

    dtw_score = dtw.distance(mfcc_ref, mfcc_input)

    def to_prob(arr):
        arr = np.abs(arr)
        return arr / arr.sum()

    jsd_score = jensenshannon(to_prob(mfcc_ref), to_prob(mfcc_input))
    pitch_diff = abs(get_pitch(y_ref) - get_pitch(y_input))

    emb_ref = get_embedding(y_ref, emb_processor, emb_model)
    emb_input = get_embedding(y_input, emb_processor, emb_model)
    cos_sim = np.dot(emb_ref, emb_input) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_input))

    return dtw_score, jsd_score, pitch_diff, cos_sim, mfcc_ref, mfcc_input

# --- 3. Jitter / Shimmer ---

def get_jitter_shimmer_scalar(path):
    """전체 평균 jitter/shimmer (메트릭 표시용)"""
    try:
        sound = parselmouth.Sound(path)
        pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return jitter * 100, shimmer * 100
    except Exception:
        return 0.0, 0.0

def get_frame_jitter_shimmer(path, frame_duration=0.025, hop=0.010):
    """프레임별 jitter/shimmer (시각화용)"""
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

# --- 4. CTC 타임라인 ---

def get_word_timeline(y, processor, model):
    try:
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        outputs = processor.batch_decode(predicted_ids, output_word_offsets=True)

        ratio = getattr(model.config, "inputs_to_logits_ratio", 320)
        time_offset = ratio / 16000

        timeline = []
        for d in outputs.word_offsets[0]:
            timeline.append({
                "Word": d["word"],
                "Start": round(d["start_offset"] * time_offset, 2),
                "End": round(d["end_offset"] * time_offset, 2)
            })
        return timeline
    except Exception as e:
        st.warning(f"타임라인 분석 실패: {e}")
        return []

# --- 5. 시각화 ---

def plot_waveform_analysis(y_ref, y_input, input_path, timeline, sr=16000, word=""):
    """파형 + CTC 세그먼트 + 프레임별 Jitter/Shimmer 통합 시각화"""
    t_ref = np.linspace(0, len(y_ref) / sr, len(y_ref))
    t_input = np.linspace(0, len(y_input) / sr, len(y_input))

    js_times, jitters, shimmers = get_frame_jitter_shimmer(input_path)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.06,
        subplot_titles=(
            f"🎵 파형 비교 + CTC 세그먼트 — '{word}'",
            "📳 Jitter (목소리 떨림, 프레임별)",
            "🔊 Shimmer (음량 불안정, 프레임별)"
        )
    )

    # 파형
    fig.add_trace(go.Scatter(
        x=t_ref, y=y_ref, mode="lines", name="Native",
        line=dict(color="rgba(100,149,237,0.6)", width=1),
        fill="tozeroy", fillcolor="rgba(100,149,237,0.15)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_input, y=y_input, mode="lines", name="Input",
        line=dict(color="rgba(255,99,71,0.7)", width=1),
        fill="tozeroy", fillcolor="rgba(255,99,71,0.15)"
    ), row=1, col=1)

    # CTC 세그먼트
    y_max = max(np.abs(y_ref).max(), np.abs(y_input).max()) * 1.1
    seg_colors = [
        "rgba(255,215,0,0.15)", "rgba(144,238,144,0.15)",
        "rgba(173,216,230,0.15)", "rgba(255,160,122,0.15)",
        "rgba(221,160,221,0.15)"
    ]
    for i, seg in enumerate(timeline):
        fig.add_vrect(
            x0=seg["Start"], x1=seg["End"],
            fillcolor=seg_colors[i % len(seg_colors)],
            opacity=0.6, line_width=1,
            line_color="rgba(255,255,255,0.3)",
            row=1, col=1
        )
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

    # Jitter
    fig.add_trace(go.Scatter(
        x=js_times, y=jitters, mode="lines", name="Jitter (%)",
        line=dict(color="rgba(255,165,0,0.9)", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,165,0,0.2)"
    ), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash",
                  line_color="rgba(255,255,255,0.4)",
                  annotation_text="정상 상한(1%)",
                  annotation_position="top right",
                  row=2, col=1)
    for seg in timeline:
        for t in [seg["Start"], seg["End"]]:
            fig.add_vline(x=t, line_dash="dot",
                          line_color="rgba(255,255,255,0.2)",
                          row=2, col=1)

    # Shimmer
    fig.add_trace(go.Scatter(
        x=js_times, y=shimmers, mode="lines", name="Shimmer (%)",
        line=dict(color="rgba(0,206,209,0.9)", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,206,209,0.2)"
    ), row=3, col=1)
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

def plot_mfcc(mfcc_ref, mfcc_input, jsd_score):
    def to_prob(arr):
        arr = np.abs(arr)
        return arr / arr.sum()

    fig, axes = plt.subplots(1, 3, figsize=(6, 2))

    axes[0].bar(range(13), mfcc_ref, alpha=0.5, label="Native", color="blue")
    axes[0].bar(range(13), mfcc_input, alpha=0.5, label="Input", color="red")
    axes[0].set_title("MFCC Distribution")
    axes[0].legend()

    delta = mfcc_ref - mfcc_input
    axes[1].plot(delta, marker='o', color='purple')
    axes[1].axhline(0, color='black', ls='--')
    axes[1].set_title("Delta (native - input)")

    axes[2].plot(to_prob(mfcc_ref), label="native", color="blue")
    axes[2].plot(to_prob(mfcc_input), label="input", color="red")
    axes[2].fill_between(range(13), to_prob(mfcc_ref), to_prob(mfcc_input),
                         alpha=0.3, color="purple")
    axes[2].set_title(f"JSD Distribution (JSD={jsd_score:.2f})")
    axes[2].legend()

    st.pyplot(fig)

# --- 6. LLM 피드백 ---

def get_llm_feedback(dtw_score, jsd_score, pitch_diff, cos_sim,
                     jitter, shimmer, timeline, mfcc_ref, mfcc_input, word):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    delta = (mfcc_ref - mfcc_input).tolist()

    prompt = f"""
당신은 '말모리'의 AI 한국어(영어) 발음 교정 전문가입니다.
아래 수치를 기반으로 맞춤 피드백을 한국어로 주세요.

분석 단어: {word}

[음성 분석 지표]
- DTW 편차: {dtw_score:.2f} (100 이하 양호)
- JSD 유사도: {jsd_score:.2f} (0에 가까울수록 유사)
- Pitch 차이: {pitch_diff:.1f} Hz (50Hz 이상이면 억양 차이)
- 코사인 유사도: {cos_sim:.2f} (1에 가까울수록 유사)
- Jitter(떨림): {jitter:.3f}%
- Shimmer(음량 안정도): {shimmer:.3f}%
- 타임라인: {timeline}

MFCC Delta: {[round(v, 2) for v in delta]}
(양수 = 해당 대역 에너지 부족 / 0-3번: 저주파 모음 / 8-12번: 고주파 자음)

다음 형식으로 피드백 주세요:
1. 전반적인 평가 (발음 완성도 점수 포함)
2. 잘된 점
3. 개선이 필요한 부분 (MFCC Delta 기반 구체적으로)
4. 타임라인 기반 음절 속도 피드백
5. 교정 팁
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 7. 사이드바 ---

with st.sidebar:
    st.header("앱 작동 원리")
    st.markdown("""
    1. **기준 음성** - gTTS로 원어민 발음 생성
    2. **업로드** - 내 발음 녹음 후 업로드
    3. **음성 인식** - Whisper 단어 자동 인식
    4. **벡터화** - MFCC로 음성 벡터 변환
    5. **임베딩** - wav2vec 딥러닝 특징 추출
    6. **비교** - DTW 편차 + JSD 분포 차이
    7. **발성 분석** - 프레임별 Jitter/Shimmer
    8. **타임라인** - CTC 기반 구간 분석
    9. **AI 피드백** - LLM 맞춤 솔루션
    """)
    st.header("학습 단어 목록")
    st.write(", ".join(WORDS))

# --- 8. 메인 ---

def main():
    st.title("🎙️ 말모리 : AI 발음 편차 분석기")
    st.caption("고전 음향 신호 처리와 딥러닝을 결합한 정밀 발음 피드백")

    generate_reference(WORDS)
    whisper_model = load_whisper()
    ctc_processor, ctc_model = load_ctc_model()
    emb_processor, emb_model = load_embedding_model()

    uploaded = st.file_uploader("발음 녹음 업로드 (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

    if not uploaded:
        return

    suffix = "." + uploaded.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
         tmp.write(uploaded.read())
         orig_path = tmp.name

# wav로 변환 (parselmouth용)
    y_wav, sr_wav = librosa.load(orig_path, sr=16000)
    wav_path = orig_path.replace(suffix, ".wav")
    sf.write(wav_path, y_wav, sr_wav)
    tmp_path = wav_path

    with st.spinner("음성 인식 중..."):
        result = whisper_model.transcribe(tmp_path, language="en")
        recognized = result["text"].strip().lower()

    st.write(f"**인식된 발음:** {recognized}")

    matched = next((w for w in WORDS if w in recognized or recognized.strip() == w), None)
    if not matched:
        st.warning(f"'{recognized}' 는 학습 목록에 없는 단어예요. 다시 녹음해보세요.")
        return

    st.write(f"**비교 기준 단어:** {matched}")

    y_ref = load_audio(f"reference/{matched}.mp3")
    y_input = load_audio(tmp_path)

    with st.spinner("편차 분석 중..."):
        dtw_score, jsd_score, pitch_diff, cos_sim, mfcc_ref, mfcc_input = compare(
            y_ref, y_input, emb_processor, emb_model
        )
        jitter, shimmer = get_jitter_shimmer_scalar(tmp_path)
        timeline = get_word_timeline(y_input, ctc_processor, ctc_model)

    # 메트릭
    cols = st.columns(6)
    cols[0].metric("DTW 편차", f"{dtw_score:.1f}")
    cols[1].metric("JSD 유사도", f"{jsd_score:.3f}")
    cols[2].metric("Pitch 차이", f"{pitch_diff:.1f} Hz")
    cols[3].metric("코사인 유사도", f"{cos_sim:.2f}")
    cols[4].metric("Jitter(떨림)", f"{jitter:.3f}%")
    cols[5].metric("Shimmer(안정)", f"{shimmer:.3f}%")

    # 통합 파형 시각화
    st.markdown("---")
    with st.spinner("파형 시각화 중..."):
        wave_fig = plot_waveform_analysis(y_ref, y_input, tmp_path, timeline, word=matched)
    st.plotly_chart(wave_fig, width='stretch')

    # AI 피드백
    st.markdown("---")
    with st.spinner("AI 피드백 생성 중..."):
        feedback = get_llm_feedback(
            dtw_score, jsd_score, pitch_diff, cos_sim,
            jitter, shimmer, timeline, mfcc_ref, mfcc_input, matched
        )
    st.markdown("### 🤖 AI 발음 피드백")
    st.write(feedback)

    # MFCC 그래프
    st.markdown("---")
    plot_mfcc(mfcc_ref, mfcc_input, jsd_score)

if __name__ == "__main__":
    main()
