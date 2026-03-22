import os
from joblib import dump
from base_models.linear import linear_model

def train_model():
    models = {
        'linear' : linear_model
    }

    for model in models:
        model_path = 'base_models/pretrained'
        os.makedirs(model_path, exist_ok=True)
        dump(model, os.path.join(model_path, 'trained_linear_model.joblib'))