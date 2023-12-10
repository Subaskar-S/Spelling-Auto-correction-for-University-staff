from flask import Flask, request, render_template
from textdistance import Jaccard
from collections import Counter
import pandas as pd
import numpy as np
import re

class SpellChecker:
    def __init__(self, file_path='Data.txt'):
        self.words = []
        with open(file_path, 'r', encoding='utf-8') as f:
            file_name_data = f.read()
            file_name_data = file_name_data.lower()
            self.words = re.findall('\w+', file_name_data)

        self.V = set(self.words)
        self.word_freq_dict = Counter(self.words)
        self.Total = sum(self.word_freq_dict.values())
        self.probs = {k: v / self.Total for k, v in self.word_freq_dict.items()}

    def correct_spell(self, input_word):
        input_word = input_word.lower()
        if input_word in self.V:
            return [input_word]  # Return a list containing the original word
        else:
            similarities = [1 - (Jaccard(qval=2).distance(v, input_word)) for v in self.word_freq_dict.keys()]
            df = pd.DataFrame.from_dict(self.probs, orient='index').reset_index()
            df = df.rename(columns={'index': 'Word', 0: 'Prob'})
            df['Similarity'] = similarities
            output = df.sort_values(['Similarity', 'Prob'], ascending=False).head()
            return output['Word'].tolist()

app = Flask(__name__)
spell_checker = SpellChecker()

# routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spell', methods=['POST', 'GET'])
def spell():
    original_text = None
    corrected_text = None
    if request.method == 'POST':
        original_text = request.form['text']
        corrected_text = spell_checker.correct_spell(original_text)

    return render_template('index.html', original_text=original_text, corrected_text=corrected_text)

# python main
if __name__ == "__main__":
    app.run(debug=True)
