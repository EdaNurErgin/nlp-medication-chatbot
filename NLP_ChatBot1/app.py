import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "./flan-t5-medicationqa-final"
MAX_INPUT_LENGTH = 256

st.set_page_config(
    page_title="MedQuery AI — Medication Information Assistant",
    page_icon="💊",
    layout="centered"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:          #0d1117;
    --surface:     #161b22;
    --surface2:    #1c2330;
    --border:      #30363d;
    --accent:      #2dd4bf;
    --accent2:     #818cf8;
    --text:        #e6edf3;
    --muted:       #8b949e;
    --success-bg:  #0d2818;
    --success-bd:  #2ea043;
    --warn-bg:     #1a1400;
    --warn-bd:     #d29922;
    --radius:      14px;
}

/* ── Page base ── */
.stApp {
    background: var(--bg);
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 780px !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    margin-bottom: 0.5rem;
}
.hero-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    margin-bottom: 1.1rem;
    opacity: 0.85;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    background: linear-gradient(135deg, #e6edf3 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.8rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.97rem;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.8rem 0;
}

/* ── Card wrapper ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 1.2rem;
}

/* ── Form elements ── */
.stTextArea textarea,
.stTextInput input,
.stSelectbox select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.08) !important;
}

label, .stTextArea label, .stTextInput label, .stSelectbox label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}

/* ── Submit button ── */
.stFormSubmitButton button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0d1117 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stFormSubmitButton button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Answer box ── */
.answer-box {
    background: var(--success-bg);
    border: 1px solid var(--success-bd);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-top: 0.4rem;
    line-height: 1.75;
    font-size: 0.97rem;
    color: #d4f0e0;
}
.answer-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--success-bd);
    margin-bottom: 0.6rem;
}
.disclaimer {
    margin-top: 1rem;
    padding: 0.7rem 1rem;
    background: var(--warn-bg);
    border: 1px solid var(--warn-bd);
    border-radius: 8px;
    color: #e3c870;
    font-size: 0.82rem;
    line-height: 1.55;
}

/* ── Example cards ── */
.example-grid {
    display: grid;
    gap: 0.9rem;
}
.ex-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
.ex-card:hover { border-color: var(--accent2); }
.ex-num {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.35rem;
}
.ex-q {
    font-weight: 500;
    font-size: 0.94rem;
    color: var(--text);
    margin-bottom: 0.4rem;
}
.ex-meta {
    font-size: 0.8rem;
    color: var(--muted);
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.ex-tag {
    background: rgba(129,140,248,0.12);
    border: 1px solid rgba(129,140,248,0.25);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    color: var(--accent2);
}

/* ── Spinner override ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Selectbox dropdown ── */
div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading (unchanged mechanics) ────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()


# ── Inference helpers (unchanged mechanics) ────────────────────────────────
def clean_text(text):
    if text is None:
        return ""
    return str(text).strip()

def make_input(question, drug="", qtype="", section=""):
    return (
        f"Answer the medication question.\n"
        f"Drug: {clean_text(drug)}\n"
        f"Question Type: {clean_text(qtype)}\n"
        f"Section: {clean_text(section)}\n"
        f"Question: {clean_text(question)}"
    )

def ask_flan_t5(question, drug, qtype, section):
    if clean_text(question) == "":
        return None, "Please enter a medication-related question."
    input_text = make_input(question, drug, qtype, section)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            min_new_tokens=40,
            num_beams=4,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            length_penalty=1.1,
            early_stopping=True,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if answer == "":
        answer = "Insufficient information found for this question."
    return answer, None


# ── UI ─────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-pill">⚕ Powered by FLAN-T5</div>
    <h1 class="hero-title">Medication Information Assistant</h1>
    <p class="hero-sub">
        Ask any medication question and receive an AI-generated answer
        based on clinical drug information. Always consult your physician or pharmacist
        for personalised advice.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Query form ──
st.markdown('<div class="card"><div class="card-title">💬 Ask a Question</div>', unsafe_allow_html=True)

with st.form("chatbot_form"):
    question = st.text_area(
        "Your Question",
        placeholder="e.g. How does rivastigmine interact with OTC sleep medicine?",
        height=110
    )
    col1, col2 = st.columns(2)
    with col1:
        drug = st.text_input(
            "Drug Name",
            placeholder="e.g. rivastigmine"
        )
    with col2:
        qtype = st.selectbox(
            "Question Type",
            ["Interaction", "Side Effects", "Dosage", "Usage", "Precautions", "General"]
        )
    section = st.text_input(
        "Section / Topic (optional)",
        placeholder="e.g. What special precautions should I follow?"
    )
    submitted = st.form_submit_button("Generate Answer →")

st.markdown('</div>', unsafe_allow_html=True)

# ── Answer display ──
if submitted:
    with st.spinner("Generating answer…"):
        answer, error = ask_flan_t5(question, drug, qtype, section)

    if error:
        st.markdown(f"""
        <div class="disclaimer">⚠️ {error}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">✦ Model Response</div>
            {answer}
            <div class="disclaimer">
                ⚠️ <strong>Medical Disclaimer:</strong> This information is generated by an AI model
                and does not constitute medical advice. Always consult your doctor or pharmacist
                before making any decisions about your medication.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Example queries ──
st.markdown('<div class="card-title" style="padding-left:0">📋 Example Queries</div>', unsafe_allow_html=True)

examples = [
    {
        "question": "How does rivastigmine interact with OTC sleep medicine?",
        "drug": "Rivastigmine",
        "qtype": "Interaction",
        "section": "What special precautions should I follow?"
    },
    {
        "question": "What are the side effects of aspirin?",
        "drug": "Aspirin",
        "qtype": "Side Effects",
        "section": "What side effects can this medication cause?"
    },
    {
        "question": "What should I do if I miss a dose of ibuprofen?",
        "drug": "Ibuprofen",
        "qtype": "Dosage",
        "section": "What should I do if I forget a dose?"
    }
]

st.markdown('<div class="example-grid">', unsafe_allow_html=True)
for i, ex in enumerate(examples, 1):
    st.markdown(f"""
    <div class="ex-card">
        <div class="ex-num">Example {i}</div>
        <div class="ex-q">{ex['question']}</div>
        <div class="ex-meta">
            <span class="ex-tag">💊 {ex['drug']}</span>
            <span class="ex-tag">🏷 {ex['qtype']}</span>
            <span style="color:var(--muted);font-size:0.78rem">{ex['section']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style="text-align:center;margin-top:3rem;color:var(--muted);font-size:0.78rem;letter-spacing:0.04em;">
    MedQuery AI &nbsp;·&nbsp; FLAN-T5 Fine-tuned on MedicationQA &nbsp;·&nbsp;
    For informational use only
</div>
""", unsafe_allow_html=True)