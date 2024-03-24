import tempfile
from sklearn.linear_model import LogisticRegression
# https://tech.preferred.jp/ja/blog/hyperparameter-tuning-with-optuna-integration-lightgbm-tuner/ を用いることでLightgbmでの探索を簡素化できる
# 今回はscikit-learnのAPIを用いたため愚直に実装している
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import optuna
from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self, train_x:np.array, train_y:np.array, test_x:np.array, test_y:np.array):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model = None
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_x)

    @abstractmethod
    def search_best_params(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def _visualize(self, params:dict, study, output_dir:str='./image'):
        print('Hyper parameter importance:')
        search_params: list = list(params.keys())
        importances = optuna.importance.get_param_importances(
            study=study,
            params=search_params)
        for k, v in importances.items():
            print(f'{k}: {v}')

        # .show()をおこなうとその場で可視化する。jupyterの場合は.show()のほうも利用を検討すべき。
        fig = optuna.visualization.plot_param_importances(study)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.write_html(f'{output_dir}/param_importance.html')
        fig.write_image(f'{output_dir}/param_importance.png')

class OptunaLogisticRegression(BaseModel):

    def __init__(self, train_x:np.array, train_y:np.array, test_x:np.array, test_y:np.array):
        super().__init__(train_x, train_y, test_x, test_y)
        self.best_params = None

    def search_best_params(self):

        def objective(trial):
            """
            本メソッド名・引数名で定義する。
            paramsにはkeyにパラメータ名, valueにはそのパラメータ名に対応した範囲を定義する。
            ”trial_suggest_”の第一引数にパラメータ名、第二引数で最小値・第三引数で最大値を指定する。詳細は公式リファレンス参照。
            """
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2),
                'max_iter': trial.suggest_int('max_iter', 500, 2000, step=100)
            }
            model = LogisticRegression(**params)
            sk = StratifiedKFold(n_splits=3)
            train_x = self.scaler.transform(self.train_x)
            score = cross_val_score(model, train_x, self.train_y, cv=sk)
            accuracy = score.mean()

            return accuracy
        
        # create_study: 目的関数の最適化
        # 精度によってmaximize/minimizeを変更する.
        # RMSEなどはminimize
        study = optuna.create_study(direction='maximize')
        # n_trials: 試行回数を定義する。第一引数はobjectiveメソッドを指定する。
        study.optimize(objective, n_trials=10)
        # 探索結果はstudyに格納される。
        self.best_params = study.best_params
        print(f"study.best_params:{study.best_params}")
        print(f"study.best_value:{study.best_value}")

        # 以下でハイパーパラメータの重要度を算出できる。
        self._visualize(self.best_params, study, output_dir='./image_lr')

    def train(self):
        self.model = LogisticRegression(random_state=42, **self.best_params)
        train_x = self.scaler.transform(self.train_x)
        self.model.fit(train_x, self.train_y)

    def predict(self):
        test_x = self.scaler.transform(self.test_x)
        pred = self.model.predict(test_x)
        accuracy = accuracy_score(self.test_y, pred)
        print(f"LR: Optuna average score: {accuracy:.3f}")


class OptunaLGbMClassifier(BaseModel):

    def __init__(self, train_x:np.array, train_y:np.array, test_x:np.array, test_y:np.array):
        super().__init__(train_x, train_y, test_x, test_y)
        self.best_params = None

    def search_best_params(self):

        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 5, 100),
                'min_child_weight': trial.suggest_uniform('min_child_weight', 0.001, 0.1),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }
            # verbose=-1を指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
            model = lgb.LGBMClassifier(n_estimators=10000, verbose=-1, **params)
            # デフォルトのcross_val_scoreではearly_stoppingを適用できないため簡易的に実装
            score = self._cross_val_score(model, self.train_x, self.train_y)
            accuracy = score.mean()
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        # n_trials: 試行回数を定義する。第一引数はobjectiveメソッドを指定する。
        study.optimize(objective, n_trials=100)
        # 探索結果はstudyに格納される。
        self.best_params = study.best_params
        print(f"study.best_params:{study.best_params}")
        print(f"study.best_value:{study.best_value}")

        # 以下でハイパーパラメータの重要度を算出できる。
        self._visualize(self.best_params, study, output_dir='./image_lgbm')

    def train(self):
        self.model = lgb.LGBMClassifier(random_state=42, verbose=-1, **self.best_params)
        self.model.fit(self.train_x, self.train_y)

    def predict(self):
        pred = self.model.predict(self.test_x)
        accuracy = accuracy_score(self.test_y, pred)
        print(f"LGBM: Optuna average score: {accuracy:.3f}")

    def _cross_val_score(self, model, X: np.array, y: np.array) -> np.array:
        acc = np.array([])
        cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        for train_index, valid_index in cv.split(X, y):
            train_x, valid_x = X[train_index], X[valid_index]
            train_y, valid_y = y[train_index], y[valid_index]
            model.fit(train_x,
                      train_y,
                      eval_set=[(valid_x, valid_y)],
                      callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True), lgb.log_evaluation(0)],
                      )
            pred_y = model.predict(valid_x)
            ac = accuracy_score(valid_y, pred_y)
            acc = np.append(acc, ac)
        return acc


def main():
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target)

    lgbm_model: BaseModel = OptunaLGbMClassifier(train_x, train_y, test_x, test_y)
    logistic_model: BaseModel = OptunaLogisticRegression(train_x, train_y, test_x, test_y)
    models = [lgbm_model, logistic_model]
    for m in models:
        m.search_best_params()
        m.train()
        m.predict()

if __name__ == '__main__':
    main()