import pandas as pd
import numpy as np

from pycaret.classification import *
from pycaret.datasets import get_data

def main():
    data = get_data('iris')
    #s = setup(data, target='species', session_id=42)
    #print(s)

    from pycaret.classification import ClassificationExperiment
    exp = ClassificationExperiment()
    print(type(exp))
    exp.setup(data, target='species', session_id=123)

    # compare
    best = exp.compare_models()
    print(best)

    """
    # train
    best = compare_models()

    # evaluate training models
    evaluate_model(best)

    # predict
    pred_holdout = predict_model(best)

    # predict on new data
    new_data = data.copy().drop('species', axis=1)
    predictions = predict_model(best, data=new_data)

    # save model
    save_model(best, 'best_pipeline')
    """

if __name__ == "__main__":
    main()
