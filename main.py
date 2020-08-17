import pandas as pd
import numpy as np
from metaflow import FlowSpec, step
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from fe_initial_charactor import FeatureInitialCharactor
from fe_intersection_words import FeatureIntersectionWords
from fe_named_entity import FeatureNamedEntity
from fe_non_intersection_words import FeatureNonIntersectionWords
from fe_number import FeatureNumber
from fe_w2v import FeatureW2V
from fe_pretrained_bert import FeaturePretrainedBert
from lgbm_classifer import LightGBMClassifer


class LGBM_RTE(FlowSpec):

    @step
    def start(self):
        print("workflow starting.")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """ データのロード """
        df = pd.read_csv("data/entail_evaluation_set.txt", sep=" ", index_col=0, header=None, names=["id", "cat", "label", "t1", "t2"])
        mapping = {
                '×': '×',
                '△': '×',
                '○': '○',
                '◎': '○'
            }
        df.label = df.label.map(mapping)

        # データの抽出
        self.t1 = df.t1.values
        self.t2 = df.t2.values
        labels = df.label.values

        # ラベルエンコーディング
        le = LabelEncoder()
        self.labels = le.fit_transform(labels)

        self.next(self.number, self.named_entity, self.intersection_words, 
                    self.initial_charactor, self.w2v, self.non_intersection_words, self.pretrained_bert_embedding)

    # Feature Engineering
    @step
    def number(self):
        num = FeatureNumber()
        
        self.f1 = []
        self.labels = self.labels
        for sent1, sent2 in zip(self.t1, self.t2):
            self.f1.append(num.get(primary_text=sent1, secondary_text=sent2))

        self.next(self.train)

    @step
    def named_entity(self):
        named_entity = FeatureNamedEntity()
        
        self.f2 = []
        for sent1, sent2 in zip(self.t1, self.t2):
            self.f2.append(named_entity.get(primary_text=sent1, secondary_text=sent2))
        self.next(self.train)

    @step
    def intersection_words(self):
        intersection_words = FeatureIntersectionWords()

        self.f3 = []
        for sent1, sent2 in zip(self.t1, self.t2):
            self.f3.append(intersection_words.get(primary_text=sent1, secondary_text=sent2))

        self.next(self.train)

    @step
    def initial_charactor(self):
        initial_charactor = FeatureInitialCharactor()

        self.f4 = []
        for sent1, sent2 in zip(self.t1, self.t2):
            self.f4.append(initial_charactor.get(primary_text=sent1, secondary_text=sent2))

        self.next(self.train)

    @step
    def w2v(self):
        w2v = FeatureW2V()

        self.f5 = []
        for sent1, sent2 in zip(self.t1, self.t2):
            self.f5.append(w2v.get(primary_text=sent1, secondary_text=sent2))

        self.next(self.train)

    @step
    def non_intersection_words(self):
        non_intersection_words = FeatureNonIntersectionWords()

        self.f7 = []
        for sent1, sent2 in zip(self.t1, self.t2):
            self.f7.append(non_intersection_words.get(primary_text=sent1, secondary_text=sent2))

        self.next(self.train)

    @step
    def pretrained_bert_embedding(self):
        pretrained_bert_embedding = FeaturePretrainedBert()

        self.f8 = np.empty((0,768), float)
        for sent1, sent2 in zip(self.t1, self.t2):
            embedding = pretrained_bert_embedding.get(primary_text=sent1, secondary_text=sent2)
            embedding = embedding.reshape(1, 768)
            self.f8 = np.append(self.f8, embedding, axis=0)

        self.next(self.train)

    # Training

    @step
    def train(self, inputs):
        f1 = inputs.number.f1
        f2 = inputs.named_entity.f2
        f3 = inputs.intersection_words.f3
        f4 = inputs.initial_charactor.f4
        f5 = inputs.w2v.f5
        f7 = inputs.non_intersection_words.f7
        f8 = inputs.pretrained_bert_embedding.f8

        data = pd.DataFrame({'f1': f1,
                            'f2': f2,
                            'f3': f3,
                            'f4': f4,
                            'f5': f5,
                            'f7': f7
                            })

        embedding = pd.DataFrame(f8)

        data = pd.merge(data, embedding, right_index=True, left_index=True, how="left")

        label = inputs.number.labels

        train_df, self.test_df, y_train_s, self.y_test_s = train_test_split(
           data, label, test_size=0.1, random_state=0, stratify=label
        )

        self.train_df, self.validation_df, self.y_train_s, self.y_validation_s = train_test_split(
           train_df, y_train_s, test_size=0.1, random_state=0
        )

        classifer = LightGBMClassifer()
        self.model = classifer.train(X_train=self.train_df, y_train=self.y_train_s, X_valid=self.validation_df, y_valid=self.y_validation_s)

        self.next(self.evaluation)

    @step
    def evaluation(self):
        y_pred = self.model.predict(self.test_df, num_iteration=self.model.best_iteration)
        predict = [0 if i < 0.5 else 1 for i in y_pred]
        y_true = self.y_test_s

        print(classification_report(y_true, predict, digits=4))
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    LGBM_RTE()
