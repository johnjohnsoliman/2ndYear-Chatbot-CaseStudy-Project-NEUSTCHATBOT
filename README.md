# NEUST Smart Handbook Chatbot

A web‑based chatbot that helps students quickly get accurate information from the NEUST Student Handbook using RAG (Retrieval-Augmented Generation).

## Features

- Ask questions about NEUST policies, admissions, grading, scholarships, and more.
- Retrieves relevant handbook passages using BM25 + TF‑IDF.
- Extracts answers with a fine‑tuned RoBERTa‑based QA model.
- Automatic fallback to pattern matching and keyword extraction.

## Project Structure

- `main.py` – Flask backend
- `templates/index.html` – frontend interface
- `handbook.pdf` – source document (NEUST Student Handbook)
- `neust-qa-bert-model/` – fine‑tuned model (download from Releases)

## Setup (for local running)

1. **Clone the repository**  
   `git clone https://github.com/johnjohnsoliman/2ndYear-Chatbot-CaseStudy-Project-NEUSTCHATBOT.git`

2. **Download the fine‑tuned model** from the [Releases](https://github.com/johnjohnsoliman/2ndYear-Chatbot-CaseStudy-Project-NEUSTCHATBOT/releases) page.  
   Extract it into the project folder as `neust-qa-bert-model/`.

3. **Create a virtual environment** (recommended)
    python -m venv venv
    source venv/bin/activate # macOS/Linux
    venv\Scripts\activate # Windows

4. **Install dependencies**  
`pip install -r requirements.txt`

5. **Run the chatbot**  
`python main.py`

6. **Open your browser** and go to `http://localhost:5000`

## Requirements

- Python 3.9 or higher
- See `requirements.txt` for packages

## Acknowledgments

- Built for the NEUST Students
- Uses Hugging Face Transformers, PyTorch, Flask, and PyMuPDF


