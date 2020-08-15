import pandas as pd
import ginza
import spacy
from spacy import displacy
import math


class FeatureW2V:
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
        """W2Vのコサイン距離の最大値の平均を取得する"""

        t1 = self.get_words(primary_text)
        t2 = self.get_words(secondary_text)

        l_similarity = []

        for w2 in t2:
            w_t2= self.nlp(w2)
            similarity = 0

            for w1 in t1:
                w_t1= self.nlp(w1)
                if similarity < w_t1.similarity(w_t2):
                    similarity = w_t1.similarity(w_t2)
                
            l_similarity.append(similarity)

        if len(l_similarity) != 0:
            f5 = sum(l_similarity) / len(l_similarity)
        else:
            f5 = 1

        return f5
