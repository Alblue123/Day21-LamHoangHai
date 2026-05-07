import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def train(
    all_params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    """

    # Tu dong dat tracking URI neu chua co
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    # Doc du lieu
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # --- BONUS 5: Canh bao lech lac du lieu ---
    dist = df_train["target"].value_counts(normalize=True).to_dict()
    print("--- Label Distribution ---")
    for label, ratio in dist.items():
        print(f"Lớp {label}: {ratio:.2%}")
        if ratio < 0.10:
            print(f"WARNING: Lớp {label} chiếm ít hơn 10% ({ratio:.2%}). Dữ liệu bị lệch!")

    # Tach dac trung
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # --- BONUS 2: Ho tro da thuat toan ---
    model_type = all_params.get("model_type", "random_forest")
    model_params = all_params.get(model_type, {})
    
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42, **model_params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42, **model_params)
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=42, **model_params)
    else:
        raise ValueError(f"Khong ho tro model_type: {model_type}")

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        model.fit(X_train, y_train)

        # Du doan va tinh chi so
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model: {model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # --- BONUS 3: Bao cao hieu suat tu dong ---
        report_text = f"MLOps Performance Report\n"
        report_text += f"========================\n"
        report_text += f"Model Type: {model_type}\n"
        report_text += f"Accuracy: {acc:.4f}\n"
        report_text += f"F1 Score: {f1:.4f}\n\n"
        report_text += "Confusion Matrix:\n"
        report_text += str(confusion_matrix(y_eval, preds)) + "\n\n"
        report_text += "Classification Report:\n"
        report_text += classification_report(y_eval, preds)

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        # Luu metrics JSON (Bonus 5: them distribution)
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc, 
                "f1_score": f1,
                "label_distribution": {str(k): float(v) for k, v in dist.items()}
            }, f)

        # Luu mo hinh
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        all_params = yaml.safe_load(f)
    train(all_params)
