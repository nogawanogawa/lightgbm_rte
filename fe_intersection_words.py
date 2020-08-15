import pandas as pd
import ginza
import spacy
from spacy import displacy

class FeatureIntersectionWords:
    def __init__(self):
        super().__init__()
        self.hinsi = ["NOUN", "VERB", "ADJ", "ADV"]
        self.nlp = spacy.load('ja_ginza')


    def get_words(self, text: str):
        """内容語の抽出"""
        doc = self.nlp(text)

        result = []

        for sent in doc.sents:
            for token in sent:
                if token._.ne == "" and token.pos_ in self.hinsi:
                    result.append(token.lemma_)

        return result


    def get(self, primary_text: str, secondary_text: str) -> float:
        """内容語の一致率を取得する"""

        t1 = self.get_words(primary_text)
        t2 = self.get_words(secondary_text)

        count = 0

        for word in t2:
            if word not in t1:
                count += 1

        if len(t2) != 0:
            f3 = count / len(t2)
        else :
            f3 = 0

        return f3