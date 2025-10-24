This is one of my initial attempts at developing LLM-based apps. It is intended to be used as an academic assistant that answers your questions based on a paper you uploaded.

## Quickstart
### 1. Clone the repository

```bash
git clone https://github.com/ufukcemcakir/academic-chat.git
cd academic-chat
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Set your HuggingFace API Key

```env
HF_TOKEN = "your_token"
```
Generate your token by creating a HugginfFace account if you don't have any, go to settings and "access tokens".
### 4. Run the app

```bash
streamlit run app.py
```
It will launch on http://localhost:8501.

## How It Works

Text Extraction: pdfplumber extracts text from each page.

Chunking: The paper is split into 500-character chunks with 100-character overlap.

Embedding: Chunks are embedded using all-MiniLM-L6-v2.

Search: FAISS retrieves the most relevant chunks for the user's question.

LLM Prompting: A prompt is constructed with the retrieved context and sent to the Qwen model hosted on Hugging Face.

Answer: The generated answer is displayed in the UI.

## Screenshot
<img width="1895" height="670" alt="Ekran görüntüsü 2025-10-24 214414" src="https://github.com/user-attachments/assets/1e63fb3b-86ed-4d47-ad21-a2be58f65e85" />


## What this app lacks

My goal was to familiarize with developing and deploying micro-scale LLM apps. I wanted to get the app up and running, so I simplified a lot of things.

- I used Hugging Face's Inference Provider, which gives you a limited amount of API calls per day.
- It only handles one PDF per session.
- It assumes that you upload a text-only PDF.
- There is no caching or database storage, everything runs in memory.
- Because of my compute limitations, I used a 1.7B paramaeter model. Although it is very successful for a model of its size, it may halucinate if asked too many questions.

## Author
Ufuk Cem Cakir.
Github: @ufukcemcakir
