# Import libraries

import argparse
import glob
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# define functions
def main(args):
    # Enable autologging
    mlflow.sklearn.autolog()

    # Log parameters manually for visibility
    mlflow.log_param('training_data', args.training_data)
    mlflow.log_param('reg_rate', args.reg_rate)

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # Optionally, log metrics manually (accuracy, auc)
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    y_scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)
    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('auc', auc)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    """
    Splits the input DataFrame into train and test sets for features and labels.

    Args:
        df (pandas.DataFrame): The input DataFrame containing features and target.

    Returns:
        X_train, X_test, y_train, y_test: Split feature and label arrays.
    """
    X = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
            'SerumInsulin','BMI','DiabetesPedigree','Age']].values
    y = df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)
    return model


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
