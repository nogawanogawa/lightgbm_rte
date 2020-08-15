import pandas as pd
import ginza
import spacy
from spacy import displacy


class FeatureInitialCharactor:
    def __init__(self):
        super().__init__()
        self.hinsi = ["NOUN", "VERB", "ADJ", "ADV"]
        self.nlp = spacy.load('ja_ginza')

    def get_words(self, text: str):
        """内容語の先頭文字の抽出"""
        doc = self.nlp(text)

        result = []

        for sent in doc.sents:
            for token in sent:
                if token._.ne == "" and token.pos_ in self.hinsi:
                    result.append(token.orth_[0])

        return result


    def get(self, primary_text: str, secondary_text: str) -> float:
        """内容語の先頭文字の一致率を取得する"""

        t1 = self.get_words(primary_text)
        t2 = self.get_words(secondary_text)

        count = 0

        for word in t2:
            if word not in t1:
                count += 1
        if len(t2) != 0:
            f4 = count / len(t2)
        else:
            f4 = 0

        return f4
