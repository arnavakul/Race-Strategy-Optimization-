import os
import pickle
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "strategy_type_dataset.parquet"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "api",
    "models",
    "saved_models",
    "strategy_type_predictor.pkl"
)

def load_dataset():
    
    df = pd.read_parquet(
        DATASET_PATH
    )
    
    return df

def prepare_features(df):
    
    X = df.drop(
        columns = [
            "StrategyType"
        ]
    )
    
    y = df[
        "StrategyType"
    ]
    
    X = pd.get_dummies(
        X
    )
    
    return X, y 

def split_data(X,y):
    return train_test_split(
        X,
        y,
        test_size=0.8,
        random_state=42,
        stratify=y
    )

def train_model(
    X_train,
    y_train
):
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state= 42
    )
    
    model.fit(
        X_train,
        y_train
    )
    
    return model

def evaluate_model(
    model,
    X_test,
    y_test
):
    
    predictions = model.predict(
        X_test
    )
    
    accuracy = accuracy_score(
        y_test,
        predictions
    )
    
    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )
    
    print(
        "\nClassification Report\n"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )
    
    print(
        "\nConfusion Matrix\n"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

def save_model(model):
    
    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            model,
            f
        )

    print(
        f"\nModel Saved:\n"
        f"{MODEL_PATH}"
    )

if __name__ == "__main__":

    df = load_dataset()

    X, y = prepare_features(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(
        model
    )