
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from dtaidistance import dtw
from gtts import gTTS
import whisper
import os
import tempfile

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource  
def load_wav2vec():
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

def generate_reference(words):
    os.makedirs("reference", exist_ok=True)
    for word in words:
        path = f"reference/{word}.mp3"
        if not os.path.exists(path):
            gTTS(word, lang="en", tld="com").save(path)

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return librosa.util.normalize(y)

def get_embedding(y, processor, model):
    import torch
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_mfcc(y, sr=16000):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)

def get_pitch(y, sr=16000):
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    vals = pitch[pitch > 0]
    return vals.mean() if vals.size > 0 else 0

def compare(y_ref, y_input, processor, model):
    emb_ref = get_embedding(y_ref, processor, model)
    emb_input = get_embedding(y_input, processor, model)
    mfcc_ref = get_mfcc(y_ref)
    mfcc_input = get_mfcc(y_input)
    dtw_score = dtw.distance(mfcc_ref, mfcc_input)
    def to_prob(arr):
        arr = np.abs(arr)
        return arr / arr.sum()
    jsd_score = jensenshannon(to_prob(mfcc_ref), to_prob(mfcc_input))
    pitch_diff = abs(get_pitch(y_ref) - get_pitch(y_input))
    cos_sim = np.dot(emb_ref, emb_input) / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_input))
    return dtw_score, jsd_score, pitch_diff, cos_sim, mfcc_ref, mfcc_input

def get_feedback(jsd_score, dtw_score, pitch_diff):
    if jsd_score < 0.2:
        return "✅ 원어민 발음과 유사합니다"
    elif jsd_score < 0.4:
        if pitch_diff > 50:
            return "⚠️ 억양을 더 자연스럽게 조절해보세요"
        return "⚠️ 전반적인 발음이 조금 다릅니다"
    else:
        return "❌ 발음 교정이 필요합니다. 원어민 발음을 다시 들어보세요"

def main():
    st.title("🎙️ 발음 편차 분석기")
    st.caption("원어민 발음과 내 발음의 차이를 분석합니다")

    words = [
        "orange", "apple", "banana", "computer", "internet",
        "chocolate", "camera", "energy", "coffee", "television"
    ]

    generate_reference(words)
    whisper_model = load_whisper()
    processor, wav2vec_model = load_wav2vec()

    uploaded = st.file_uploader("발음 녹음 업로드 (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

    if uploaded:
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("음성 인식 중..."):
            result = whisper_model.transcribe(tmp_path)
            recognized = result["text"].strip().lower()

        st.write(f"**인식된 발음:** {recognized}")

        matched = None
        for word in words:
            if word in recognized:
                matched = word
                break

        if matched:
            st.write(f"**비교 기준 단어:** {matched}")
            y_ref = load_audio(f"reference/{matched}.mp3")
            y_input = load_audio(tmp_path)

            with st.spinner("편차 분석 중..."):
                dtw_score, jsd_score, pitch_diff, cos_sim, mfcc_ref, mfcc_input = compare(
                    y_ref, y_input, processor, wav2vec_model
                )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("DTW 편차", f"{dtw_score:.2f}")
            col2.metric("JSD 유사도", f"{jsd_score:.2f}")
            col3.metric("Pitch 차이", f"{pitch_diff:.1f} Hz")
            col4.metric("코사인 유사도", f"{cos_sim:.2f}")

            st.subheader(get_feedback(jsd_score, dtw_score, pitch_diff))

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(range(13), mfcc_ref, alpha=0.7, label="원어민", color="blue")
            axes[0].bar(range(13), mfcc_input, alpha=0.7, label="입력", color="red")
            axes[0].set_title("MFCC 분포 비교")
            axes[0].legend()
            axes[1].plot(mfcc_ref - mfcc_input, color="purple")
            axes[1].axhline(0, color="black", linestyle="--")
            axes[1].set_title("편차 (원어민 - 입력)")
            st.pyplot(fig)

        else:
            st.warning(f"'{recognized}' 는 목록에 없는 단어예요. 다시 녹음해보세요")

if __name__ == "__main__":
    main()
