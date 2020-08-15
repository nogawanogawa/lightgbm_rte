import lightgbm as lgb
import pandas as pd

class LightGBMClassifer:
    def __init__(self):
        # LightGBMのハイパーパラメータを設定
        self.lgb = lgb.LGBMClassifier()    

    def train(self, X_train, y_train, X_valid, y_valid):
        train_data = lgb.Dataset(X_train, label=y_train)
        validation_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        params = {
          'metric': 'gbdt',      # GBDTを指定
          'objective': 'binary',        # 多クラス分類を指定
          'metric': "binary_logloss",   # 多クラス分類の損失（誤差）
          'learning_rate': 0.05,        # 学習率
          'num_leaves': 21,             # ノードの数
          'min_data_in_leaf': 3,        # 決定木ノードの最小データ数
          'early_stopping_rounds':100,  # 
          'num_iteration': 1000}        # 予測器(決定木)の数:イタレーション

        self.model = lgb.train(params,
               train_set=train_data,
               valid_sets=validation_data,
               num_boost_round=10000,
               early_stopping_rounds=100,
               verbose_eval=50)
        
        return self.model

