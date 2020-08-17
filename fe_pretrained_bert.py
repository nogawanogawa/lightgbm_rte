from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import torch

class FeaturePretrainedBert:
    def __init__(self):
        super().__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.model = BertForSequenceClassification.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking", # 日本語Pre trainedモデルの指定
            num_labels = 2, # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
            output_attentions = False, # アテンションベクトルを出力するか
            output_hidden_states = True, # 隠れ層を出力するか
        )
        self.model.eval()

    def get_embedding(self, text: str):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad(): # 勾配計算なし
            all_encoder_layers = self.model(tokens_tensor)

        embedding = all_encoder_layers[1][-2].numpy()[0]
        result = np.mean(embedding, axis=0)

        return result


    def get(self, primary_text: str, secondary_text: str):
        text = primary_text + "[SEP]" + secondary_text
        f8 = self.get_embedding(text)

        return f8
