import datetime
import os
import subprocess
import sys
import pandas as pd
import xgboost as xgb
import hypertune
import argparse
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir.')
parser.add_argument("--dataset-data-url", dest="dataset_data_url",
                    type=str, help="Download url for the training data.")
parser.add_argument("--dataset-labels-url", dest="dataset_labels_url",
                    type=str, help="Download url for the training data labels.")
parser.add_argument("--n-estimators", dest="n_estimators",
                    default=50, type=int)
parser.add_argument("--max-depth", dest="max_depth",
                    default=5, type=int)
parser.add_argument("--learning-rate", dest="learning_rate",
                    default=0.1, type=float)
parser.add_argument("--subsample", dest="subsample",
                    default=0.5, type=float)
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

def get_data():
    logging.info("Downloading training data and labelsfrom: {}, {}".format(args.dataset_data_url, args.dataset_labels_url))
    # gsutil outputs everything to stderr. Hence, the need to divert it to stdout.
    subprocess.check_call(['gsutil', 'cp', args.dataset_data_url, 'data.csv'], stderr=sys.stdout)
    subprocess.check_call(['gsutil', 'cp', args.dataset_labels_url, 'target.csv'], stderr=sys.stdout)
    
    data   = pd.read_csv('data.csv')
    labels = pd.read_csv('target.csv')

    # drop unnamed columns
    data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    labels.drop(labels.columns[labels.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    data   = data.values
    labels = labels.values
    
    # Convert one-column 2D array into 1D array for use with XGBoost
    labels = labels.reshape((labels.size,))

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.8, random_state=42)

    return train_data, test_data, train_labels, test_labels

def train_model(train_data, train_labels):
    logging.info("Start training ...")
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        random_state=42
        )
    model.fit(train_data, train_labels)
    logging.info("Training completed")
    return model

def evaluate_model(model, test_data, test_labels):
    pred = model.predict(test_data)
    predictions = [round(value, 1) for value in pred]
    
    # evaluate predictions
    mse = mean_squared_error(test_labels, predictions)
    logging.info(f"Evaluation completed with model mean_squared_error: {mse}")

    # report metric for hyperparameter tuning
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mean_squared_error',
        metric_value=mse
    )
    return mse


train_data, test_data, train_labels, test_labels = get_data()
model = train_model(train_data, train_labels)
accuracy = evaluate_model(model, test_data, test_labels)

# GCSFuse conversion
gs_prefix = 'gs://'
gcsfuse_prefix = '/gcs/'
if args.model_dir.startswith(gs_prefix):
    args.model_dir = args.model_dir.replace(gs_prefix, gcsfuse_prefix)
    dirpath = os.path.split(args.model_dir)[0]
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

# Export the classifier to a file
gcs_model_path = os.path.join(args.model_dir, 'model.bst')
logging.info("Saving model artifacts to {}". format(gcs_model_path))
model.save_model(gcs_model_path)

logging.info("Saving metrics to {}/metrics.json". format(args.model_dir))
gcs_metrics_path = os.path.join(args.model_dir, 'metrics.json')
with open(gcs_metrics_path, "w") as f:
    f.write(f"{'mean_squared_error: {mean_squared_error}'}")