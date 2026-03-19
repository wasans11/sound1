import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

# --- 2. 핵심 분석 함수 ---

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

def get_jitter_shimmer(path):
    try:
        sound = parselmouth.Sound(path)
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return jitter * 100, shimmer * 100
    except Exception:
        return 0.0, 0.0

def get_word_timeline(y, processor, model):
    try:
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        outputs = processor.batch_decode(predicted_ids, output_word_offsets=True)

        # inputs_to_logits_ratio 없는 모델 대응
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

def compare(y_ref, y_input, emb_processor, emb_model):
    """기준 음성과 입력 음성의 전체 편차를 계산하여 반환"""
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

def generate_reference(words):
    os.makedirs("reference", exist_ok=True)
    for word in words:
        path = f"reference/{word}.mp3"
        if not os.path.exists(path):
            gTTS(word, lang="en", tld="com").save(path)

# --- 3. 시각화 ---

def plot_interactive_timeline(timeline):
    if not timeline:
        return
    df = pd.DataFrame(timeline)
    fig = px.timeline(
        df, x_start="Start", x_end="End", y="Word", color="Word",
        text="Word", title="🕒 단어 구간별 발음 타임라인",
        template="plotly_dark", height=250
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_mfcc(mfcc_ref, mfcc_input, jsd_score):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    def to_prob(arr):
        arr = np.abs(arr)
        return arr / arr.sum()

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
    axes[2].fill_between(range(13), to_prob(mfcc_ref), to_prob(mfcc_input), alpha=0.3, color="purple")
    axes[2].set_title(f"JSD Distribution (JSD={jsd_score:.2f})")
    axes[2].legend()

    st.pyplot(fig)

# --- 4. LLM 피드백 ---

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

# --- 5. 사이드바 ---

with st.sidebar:
    st.header("앱 작동 원리")
    st.markdown("""
    1. **기준 음성** - gTTS로 원어민 발음 생성
    2. **업로드** - 내 발음 녹음 후 업로드
    3. **음성 인식** - Whisper 단어 자동 인식
    4. **벡터화** - MFCC로 음성 벡터 변환
    5. **임베딩** - wav2vec 딥러닝 특징 추출
    6. **비교** - DTW 편차 + JSD 분포 차이
    7. **발성 분석** - Jitter/Shimmer
    8. **타임라인** - CTC 기반 구간 분석
    9. **AI 피드백** - LLM 맞춤 솔루션
    """)
    st.header("학습 단어 목록")
    st.write(", ".join(WORDS))

# --- 6. 메인 ---

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
        tmp_path = tmp.name

    with st.spinner("음성 인식 중..."):
        result = whisper_model.transcribe(tmp_path, language="en")
        recognized = result["text"].strip().lower()

    st.write(f"**인식된 발음:** {recognized}")

    # 단어 매칭 - 실패 시 경고 후 중단
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
        jitter, shimmer = get_jitter_shimmer(tmp_path)
        timeline = get_word_timeline(y_input, ctc_processor, ctc_model)

    # 메트릭
    cols = st.columns(6)
    cols[0].metric("DTW 편차", f"{dtw_score:.1f}")
    cols[1].metric("JSD 유사도", f"{jsd_score:.3f}")
    cols[2].metric("Pitch 차이", f"{pitch_diff:.1f} Hz")
    cols[3].metric("코사인 유사도", f"{cos_sim:.2f}")
    cols[4].metric("Jitter(떨림)", f"{jitter:.3f}%")
    cols[5].metric("Shimmer(안정)", f"{shimmer:.3f}%")

    # 타임라인
    plot_interactive_timeline(timeline)

    # AI 피드백
    with st.spinner("AI 피드백 생성 중..."):
        feedback = get_llm_feedback(
            dtw_score, jsd_score, pitch_diff, cos_sim,
            jitter, shimmer, timeline, mfcc_ref, mfcc_input, matched
        )
    st.markdown("### 🤖 AI 발음 피드백")
    st.write(feedback)

    # MFCC 그래프
    plot_mfcc(mfcc_ref, mfcc_input, jsd_score)

if __name__ == "__main__":
    main()
