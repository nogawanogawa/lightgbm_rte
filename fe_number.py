import pandas as pd
import ginza
import spacy
from spacy import displacy


class FeatureNumber:
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load('ja_ginza')

    def get_quantity(self, text: str):
        """数量の抽出"""
        doc = self.nlp(text)

        result = []
        word = ""

        for sent in doc.sents:
            for token in sent:
                if token._.ne.endswith(("QUANTITY", "TIME")):

                    # 節にまとめる
                    if token._.ne.startswith("B_") and word != "":  # 新規の数詞のとき
                        result.append(word)
                        word = ""

                    word = word + token.orth_

        if word != "":
            result.append(word)

        return result


    def get(self, primary_text: str, secondary_text: str) -> float:
        """数値の矛盾が存在するか検知する"""

        t1 = self.get_quantity(primary_text)
        t2 = self.get_quantity(secondary_text)

        f1 = 0.9

        for word in t2:
            if word not in t1:
                f1 = 0.1

        return f1