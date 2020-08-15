import pandas as pd
import ginza
import spacy
from spacy import displacy

class FeatureNamedEntity:
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load('ja_ginza')

    def get_named_entity(self, text: str):
        """固有表現の抽出"""
        doc = self.nlp(text)

        result = []
        word = ""

        for sent in doc.sents:
            for token in sent:
                if token._.ne != "": # 何らかの固有表現だったら

                    # 節にまとめる
                    if token._.ne.startswith("B_") and word != "":  # 新規の数詞のとき
                        result.append(word)
                        word = ""

                    word = word + token.orth_

        if word != "":
            result.append(word)

        return result


    def get(self, primary_text: str, secondary_text: str) -> float:
        """同じ固有表現が存在するか判定する"""

        t1 = self.get_named_entity(primary_text)
        t2 = self.get_named_entity(secondary_text)

        f2 = 0.9

        for word in t2:
            if word not in t1:
                f2 = 0.1

        return f2