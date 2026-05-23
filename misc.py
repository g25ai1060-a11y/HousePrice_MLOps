import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data():

    data = load_diabetes(
        as_frame=True
    )

    df = data.frame

    return df


def split_data(df):

    X = df.drop(
        "target",
        axis=1
    )

    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def train_model(
    model,
    X_train,
    y_train
):

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate(
    model,
    X_test,
    y_test
):

    pred = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        pred
    )

    return mse