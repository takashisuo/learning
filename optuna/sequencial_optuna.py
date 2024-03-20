from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
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

    @abstractmethod
    def search_best_params(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

class StandardLogisticRegression(BaseModel):

    def __init__(self, train_x:np.array, train_y:np.array, test_x:np.array, test_y:np.array):
        super().__init__(train_x, train_y, test_x, test_y)

    def search_best_params(self):
        pass

    def train(self):
        # 違いを見るためにCに想定外の値を設定する
        self.model = LogisticRegression(random_state=42, C=1e-4)
        self.model.fit(self.train_x, self.train_y)

    def predict(self):
        scores = cross_val_score(self.model, self.test_x, self.test_y, cv=3)
        accuracy = scores.mean()
        print(f"Standard average score: {accuracy:.3f}")


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
            score = cross_val_score(model, self.train_x, self.train_y, cv=3)
            accuracy = score.mean()

            return accuracy
        
        # create_study: 目的関数の最適化
        # 精度によってmaximize/minimizeを変更する.
        # RMSEなどはminimize
        study = optuna.create_study(direction='maximize')
        # n_trials: 試行回数を定義する。第一引数はobjectiveメソッドを指定する。
        study.optimize(objective, n_trials=100)
        # 探索結果はstudyに格納される。
        self.best_params = study.best_params
        print(f"study.best_params:{study.best_params}")
        print(f"study.best_value:{study.best_value}")

        # 以下でハイパーパラメータの重要度を算出できる。
        self._visualize(self.best_params, study)

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

    def train(self):
        self.model = LogisticRegression(random_state=42, **self.best_params)
        self.model.fit(self.train_x, self.train_y)

    def predict(self):
        scores = cross_val_score(self.model, self.test_x, self.test_y, cv=3)
        accuracy = scores.mean()
        print(f"Optuna average score: {accuracy:.3f}")

    
def main():
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)

    std_model: BaseModel = StandardLogisticRegression(train_x, train_y, test_x, test_y)
    optuna_model: BaseModel = OptunaLogisticRegression(train_x, train_y, test_x, test_y)
    models = [optuna_model, std_model]
    for m in models:
        m.search_best_params()
        m.train()
        m.predict()

if __name__ == '__main__':
    main()
