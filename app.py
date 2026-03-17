
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
from groq import Groq

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


def get_llm_feedback(dtw_score, jsd_score, pitch_diff, cos_sim, mfcc_ref, mfcc_input, word):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    delta = (mfcc_ref - mfcc_input).tolist()
    
    prompt = f"""
당신은 영어 발음 교정 전문가입니다. 아래 음향 분석 수치를 보고 한국어로 피드백을 주세요.

단어: {word}

[분석 수치]
- DTW 편차: {dtw_score:.2f} (낮을수록 유사, 보통 100 이하면 양호)
- JSD 유사도: {jsd_score:.2f} (0에 가까울수록 유사, 0.3 이상이면 차이 큼)
- Pitch 차이: {pitch_diff:.1f} Hz (50Hz 이상이면 억양 차이 있음)
- 코사인 유사도: {cos_sim:.2f} (1에 가까울수록 유사)

[MFCC 벡터 - 13개 계수, 각 주파수 대역 에너지]
- 원어민: {[round(v,2) for v in mfcc_ref.tolist()]}
- 사용자: {[round(v,2) for v in mfcc_input.tolist()]}
- Delta(원어민-사용자): {[round(v,2) for v in delta]}

Delta 값이 양수이면 사용자가 해당 주파수 대역에서 에너지가 부족한 것입니다.
낮은 계수(0-3번)는 저주파(모음), 높은 계수(8-12번)는 고주파(자음/마찰음)에 해당합니다.

다음 형식으로 피드백 주세요:
1. 전반적인 평가 (한 줄)
2. 잘된 점
3. 개선이 필요한 부분 (MFCC Delta 기반으로 구체적으로)
4. 교정 팁
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
        
with st.sidebar:
    st.header("앱 작동 원리")
    st.markdown("""
    1. **기준 음성** - gTTS로 원어민 영어 발음 생성
    2. **업로드** - 내 발음 녹음 후 업로드
    3. **음성 인식** - Whisper가 단어 자동 인식
    4. **벡터화** - MFCC로 음성을 벡터로 변환
    5. **임베딩** - wav2vec으로 딥러닝 특징 추출
    6. **비교** - DTW로 구간별 편차, JSD로 분포 차이 계산
    7. **출력** - 편차 시각화 + 발음 피드백
    8. **AI 해석** - Groq LLM이 MFCC/DTW/JSD 벡터값 기반 발음 피드백 생성
    """)
    st.header("학습 단어 목록")
    st.write("orange, apple, banana, computer, internet, chocolate, camera, energy, coffee, television")

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
            result = whisper_model.transcribe(tmp_path, language="en")
            recognized = result["text"].strip().lower()

        st.write(f"**인식된 발음:** {recognized}")

        matched = None
        for word in words:
            if word in recognized or recognized.strip() in word:
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


            with st.spinner("AI 피드백 생성 중..."):
                llm_feedback = get_llm_feedback(dtw_score, jsd_score, pitch_diff, cos_sim, mfcc_ref, mfcc_input, matched)
                st.markdown("### 🤖 AI 발음 피드백")
                st.write(llm_feedback)

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))

            def to_prob(arr):
                arr = np.abs(arr)
                return arr / arr.sum()

            axes[0].bar(range(13), mfcc_ref, alpha=0.7, label="native", color="blue")
            axes[0].bar(range(13), mfcc_input, alpha=0.7, label="input", color="red")
            axes[0].set_title("MFCC Distribution")
            axes[0].legend()

            axes[1].plot(mfcc_ref - mfcc_input, color="purple")
            axes[1].axhline(0, color="black", linestyle="--")
            axes[1].set_title("Delta (native - input)")

            axes[2].plot(to_prob(mfcc_ref), label="native", color="blue")
            axes[2].plot(to_prob(mfcc_input), label="input", color="red")
            axes[2].fill_between(range(13), to_prob(mfcc_ref), to_prob(mfcc_input), alpha=0.3, color="purple")
            axes[2].set_title(f"JSD Distribution (JSD={jsd_score:.2f})")
            axes[2].legend()

            st.pyplot(fig)
            

        else:
            st.warning(f"'{recognized}' 는 목록에 없는 단어예요. 다시 녹음해보세요")

if __name__ == "__main__":
    main()
