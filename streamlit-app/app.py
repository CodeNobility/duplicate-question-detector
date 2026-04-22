import streamlit as st
import helper
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="AI Duplicate Detector", page_icon="🧠", layout="wide")

# Custom CSS (Premium UI)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
}
.block-container {
    padding-top: 2rem;
}

.glass {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #ddd;
    margin-bottom: 30px;
}

.stTextInput>div>div>input {
    background-color: rgba(255,255,255,0.1);
    color: white;
    border-radius: 10px;
}

.stButton button {
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🧠 AI Duplicate Question Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart NLP-powered similarity checker</div>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

with col1:
    q1 = st.text_area("✏️ Question 1", height=150)

with col2:
    q2 = st.text_area("✏️ Question 2", height=150)

# Button
if st.button("🚀 Analyze Now"):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("⚠️ Please enter both questions")
    else:
        with st.spinner("🔍 AI is thinking..."):
            query = helper.query_point_creator(q1, q2)
            result = model.predict(query)[0]

        st.markdown("---")

        # Result UI
        if result:
            st.markdown("""
            <div class='glass'>
                <h2 style='color:lightgreen;'>✅ Duplicate Questions</h2>
                <p>These questions convey the same meaning.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='glass'>
                <h2 style='color:#ff4b4b;'>❌ Not Duplicate</h2>
                <p>These questions are different.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>Built with ❤️ by Prince | NLP Project</p>
""", unsafe_allow_html=True)