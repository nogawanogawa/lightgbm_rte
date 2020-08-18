from transformers import BertJapaneseTokenizer, BertForSequenceClassification, BertForSequenceClassification, AdamW, BertConfig
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from mlflow import log_metric, log_param, log_artifact

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

    def max_len(self, primary_texts, secondary_texts):
        # 最大単語数の確認
        max_len = []

        # 1文づつ処理
        for sent1, sent2 in zip(primary_texts, secondary_texts):
            token_words_1 = self.tokenizer.tokenize(sent1)
            token_words_2 = self.tokenizer.tokenize(sent2)
            token_words_1.extend(token_words_2)
            # 文章数を取得してリストへ格納
            max_len.append(len(token_words_1))
            
        max_length = max(max_len) + 3 # 最大単語数にSpecial token（[CLS], [SEP]）の+3をした値が最大単語数

        # 最大の値を確認
        return max_length

    def encode(self, primary_texts, secondary_texts, max_len):
        input_ids = []
        attention_masks = []

        # 1文づつ処理
        for x , y in zip(primary_texts, secondary_texts):
            sent= x  + "[SEP]" + y

            encoded_dict = self.tokenizer.encode_plus(
                                sent,                      
                                add_special_tokens = True, # Special Tokenの追加
                                max_length = max_len,           # 文章の長さを固定（Padding/Trancatinating）
                                pad_to_max_length = True,# PADDINGで埋める
                                return_attention_mask = True,   # Attention maksの作成
                                return_tensors = 'pt',     #  Pytorch tensorsで返す
                        )

            # 単語IDを取得     
            input_ids.append(encoded_dict['input_ids'])

            # Attention　maskの取得
            attention_masks.append(encoded_dict['attention_mask'])

        # リストに入ったtensorを縦方向（dim=0）へ結合
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    def train(self, primary_texts, secondary_texts, labels):
        max_len = self.max_len(primary_texts, secondary_texts)
        input_ids, attention_masks = self.encode(primary_texts, secondary_texts, max_len)

        # tenosor型に変換
        labels = torch.tensor(labels)

        # データセットクラスの作成
        train_dataset = TensorDataset(input_ids, attention_masks, labels)

        # データローダーの作成
        batch_size = 32
        log_param("batch_size", batch_size)

        # 訓練データローダー
        train_dataloader = DataLoader(
                    train_dataset,  
                    sampler = RandomSampler(train_dataset), # ランダムにデータを取得してバッチ化
                    batch_size = batch_size
                )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 最適化手法の設定
        lr = 2e-5
        log_param("lr", lr)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        # 学習の実行
        max_epoch = 50
        log_param("max_epoch", max_epoch)

        for epoch in range(max_epoch):
            self.model.train() # 訓練モードで実行
            train_loss = 0
            for batch in train_dataloader:# train_dataloaderはword_id, mask, labelを出力する点に注意
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                optimizer.zero_grad()
                loss, logits, _ = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            log_metric("train_loss", train_loss, step=epoch)

        return

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
