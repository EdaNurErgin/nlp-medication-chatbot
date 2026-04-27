# MedQuery AI — Medication Information Assistant

MedQuery AI is a Streamlit-based web application that leverages a fine-tuned FLAN-T5 model to answer medication-related questions using clinical drug information.

---

## 🚀 Features

* **AI-Powered Answers**: Ask questions about drug interactions, side effects, dosage, and usage.
* **Context-Aware Input**: Supports drug name, question type, and section-based queries.
* **Interactive UI**: Clean and responsive interface built with Streamlit.
* **Example Queries**: Built-in examples for easy testing.

---

## 📦 Model Download

The trained model is not included in this repository due to GitHub file size limitations.

🔗 Download the model here:
https://drive.google.com/file/d/1kSlisjbt-VNY3P8zecc5Eh5FDghK0fNM/view?usp=sharing

### After downloading:

1. Extract the zip file
2. Place the folder `flan-t5-medicationqa-final` in the project root directory

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone <your-repo-link>
cd nlp-medication-chatbot
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser:

http://localhost:8501

---

## 🧠 How It Works

The system takes user input and formats it into a structured prompt:

* Question
* Drug Name
* Question Type
* Section

This structured input is passed to a fine-tuned FLAN-T5 model, which generates a natural language answer.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit, HTML, CSS
* **Backend**: PyTorch
* **Model**: Hugging Face Transformers (FLAN-T5)
* **Libraries**: sentencepiece, protobuf, safetensors

---

## ⚠️ Disclaimer

This application is for educational purposes only.
The generated responses do not constitute medical advice.
Always consult a qualified healthcare professional.

---

## 📁 Project Structure

```
nlp-medication-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
└── flan-t5-medicationqa-final/  (download separately)
```

---

## 👩‍💻 Author

Developed as part of a Natural Language Processing course project.
