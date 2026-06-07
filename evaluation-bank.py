script = """
import json
import pathlib
import tarfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

if __name__ == "__main__":

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="/opt/ml/processing/model/")
        print("Extracted:", tar.getnames())

    model = xgb.Booster()
    model.load_model("/opt/ml/processing/model/xgboost-model.json")
    print("Model loaded")

    df = pd.read_csv("/opt/ml/processing/test/test.csv", header=None)
    y_test = df.iloc[:, 0].to_numpy()
    X_test = xgb.DMatrix(df.iloc[:, 1:])

    predictions = model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    auc_score = float(auc(fpr, tpr))
    print(f"AUC Score: {auc_score:.4f}")

    report_dict = {
        "classification_metrics": {
            "auc_score": {
                "value": auc_score,
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))

    print("Done. AUC:", round(auc_score, 4))
"""

with open("evaluation-bank.py", "w") as f:
    f.write(script)

# Verify double underscores are correct
with open("evaluation-bank.py", "r") as f:
    content = f.read()
    if '_name' in content and 'main_' in content:
        print("✅ Script saved correctly with double underscores")
    else:
        print("❌ Still wrong — check manually")
