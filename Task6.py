import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, BartTokenizer, BartForConditionalGeneration
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy
import sentencepiece
# Initialize the tokenizer and model
mt_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')
mt_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')

st.title("Multilingual Educational Content Summarizer")

# Load language models
en_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
en_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
mt_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')
mt_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')
nlp = spacy.load("en_core_web_sm")

def summarize_text(text):
    inputs = en_tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = en_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = en_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def translate_text(text, target_language='es'):
    translated = mt_model.generate(**mt_tokenizer.prepare_seq2seq_batch(src_texts=[text], return_tensors='pt'))
    return mt_tokenizer.decode(translated[0], skip_special_tokens=True)

def extract_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if token.is_stop == False and token.is_punct == False]

def calculate_readability(text):
    return {
        'Flesch Reading Ease': flesch_reading_ease(text),
        'Flesch-Kincaid Grade': flesch_kincaid_grade(text)
    }

st.title('Multilingual Educational Content Summarizer')

# Text Input
text_input = st.text_area("Input Text", "Type or paste your educational content here...")
target_language = st.selectbox("Select Target Language", ["es", "zh",])  # Add other languages if needed

if st.button('Summarize'):
    if text_input:
        # Summarize
        summary = summarize_text(text_input)
        keywords = extract_keywords(summary)
        readability_scores = calculate_readability(summary)
        translation = translate_text(summary, target_language)

        st.subheader("Original Text")
        st.write(text_input)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Keywords")
        st.write(", ".join(keywords))

        st.subheader("Readability Scores")
        st.write(readability_scores)

        st.subheader("Translation")
        st.write(translation)
    else:
        st.error("Please input some text.")