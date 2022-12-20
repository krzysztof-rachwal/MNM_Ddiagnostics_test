FROM jupyter/scipy-notebook

RUN mkdir my-model
ENV MODEL_FILE_LDA=XGB_model.joblib

RUN pip install joblib

COPY dataset.csv ./dataset.csv

COPY make_perdictions.py ./make_predictions.py

RUN python3 make_predictions.py