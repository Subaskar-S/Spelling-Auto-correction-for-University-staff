from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
import fitz  # PyMuPDF library for working with PDF files
from flask_cors import CORS

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        num_pages = doc.page_count
        for page_num in range(num_pages):
            page = doc[page_num]
            text += page.get_text()

    return text.lower()

def preprocess_text(text):
    return re.findall(r'\w+', text)

def build_vocabulary(text):
    return set(text)

def build_word_frequency_dict(text):
    return Counter(text)

def calculate_word_probabilities(word_freq_dict):
    total_words = sum(word_freq_dict.values())
    probs = {k: v / total_words for k, v in word_freq_dict.items()}
    return probs

def my_autocorrect(input_word, vocab, probs, word_freq_dict):
    input_word = input_word.lower()
    if input_word in vocab:
        return 'Your word seems to be correct'
    else:
        similarities = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in word_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Prob'], ascending=False).head()
        return output
    
@app.route('/autocorrect', methods=['POST'])
def autocorrect():
    data = request.get_json()
    input_word = data['input_word']

    # Call your autocorrect function here
    result = my_autocorrect(input_word, V, probs, word_freq_dict)

    return jsonify({'result': result.to_dict(orient='records')})


if __name__ == "__main__":
    # Example usage:
    pdf_path = r'C:\Users\Subaskar_S\AI\Datas\Data1.0.pdf'
    pdf_text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(pdf_text)
    V = build_vocabulary(preprocessed_text)
    word_freq_dict = build_word_frequency_dict(preprocessed_text)
    probs = calculate_word_probabilities(word_freq_dict)

    # Test the autocorrect function
    result = my_autocorrect('neverteless', V, probs, word_freq_dict)
    print(result)
    app.run(debug=True)
    CORS(app)


