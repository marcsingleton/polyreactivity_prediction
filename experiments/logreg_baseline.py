"""Logistic regression baseline."""

from pathlib import Path

import mlflow
import pandas as pd
import sklearn
import sklearn.metrics as metrics


def load_data(path, cols=None):
    data = pd.read_csv(path)
    y = data['split_id'].apply(lambda x: 1 if x.split('|')[1] == 'high-poly' else 0)
    X = data[cols]
    return X, y


# fmt: off
alphabet = [
    'A', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y',
    '-',
]
# fmt: on

train_path = Path('data/processed/splits/train.csv')
val_path = Path('data/processed/splits/val.csv')

# fmt: off
# These are taken from those used in the original publication:
# doi.org/10.1038/s41467-022-35276-4
# According to the IMGT numbering system, position 55 isn't considered part of CDR2. However, it is
# highly variable. Most positions here have a high level of diversity in the data set, including
# nearly all that deviate appreciably from 100% identity. The only exception is position 66.
cols = [
    '27', '28', '29', '30', '35', '37', '38',  # CDR1
    '55', '56', '57', '58', '59', '62', '63', '64', '65',  # CDR2
    '105', '106', '107', '108', '109', '110', '111', '111A', '111B', '111C', '111D',  # CDR3.1
    '112E', '112D', '112C', '112B', '112A', '112', '113', '114', '115', '116', '117',  # CDR3.2
]
# fmt: on

experiment_name = 'logreg_baseline'
model_name = 'logreg_baseline'
C = 1.0  # Regularization strength
max_iter = 1000

if __name__ == '__main__':
    X_train, y_train = load_data(train_path, cols)
    X_val, y_val = load_data(val_path, cols)

    categories = [alphabet.copy() for _ in cols]
    encoder = sklearn.preprocessing.OneHotEncoder(categories=categories, handle_unknown='error')

    X_train = encoder.fit_transform(X_train)
    X_val = encoder.transform(X_val)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        model = sklearn.linear_model.LogisticRegression(C=C, max_iter=max_iter)
        model = model.fit(X_train, y_train)

        mlflow.log_param('C', C)
        mlflow.log_param('max_iter', max_iter)
        mlflow.sklearn.log_model(model, name=model_name, serialization_format='skops')

        for split_name, X, y_true in [('train', X_train, y_train), ('val', X_val, y_val)]:
            y_pred = model.predict(X)
            y_score = model.predict_proba(X)[:, 1]  # Slice class 1 probability

            acc = metrics.accuracy_score(y_true, y_pred)
            mlflow.log_metric(f'acc_{split_name}', acc)

            recall = metrics.recall_score(y_true, y_pred)
            mlflow.log_metric(f'recall_{split_name}', recall)

            precision = metrics.precision_score(y_true, y_pred)
            mlflow.log_metric(f'precision_{split_name}', precision)

            roc_auc = metrics.roc_auc_score(y_true, y_score)
            mlflow.log_metric(f'roc_auc_{split_name}', roc_auc)

            precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
            pr_auc = metrics.auc(recall, precision)
            mlflow.log_metric(f'pr_auc_{split_name}', pr_auc)
