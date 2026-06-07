import os
import numpy as np
import pandas as pd

if __name__ == "__main__":

    base_dir = "/opt/ml/processing"

    # =============================
    # 1. Load Data
    # =============================
    df = pd.read_csv(f"{base_dir}/input/bank-full.csv", sep=";")

    # =============================
    # 2. Convert target variable
    # =============================
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    # =============================
    # 3. Handle categorical columns
    # =============================
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    df = pd.get_dummies(df, columns=categorical_cols, dtype=np.float32)

    # =============================
    # 4. Separate label & features
    # =============================
    y = df["y"]
    X = df.drop(columns=["y"])

    # =============================
    # 5. Convert to numpy (CRITICAL FIX)
    # =============================
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().reshape(-1, 1).astype(np.float32)

    # Combine → label MUST be first column
    data = np.concatenate((y, X), axis=1)

    # =============================
    # 6. Shuffle data
    # =============================
    np.random.shuffle(data)

    # =============================
    # 7. Train / Validation / Test split
    # =============================
    train, validation, test = np.split(
        data,
        [int(0.7 * len(data)), int(0.85 * len(data))]
    )

    # =============================
    # 8. Convert back to DataFrame
    # =============================
    train = pd.DataFrame(train)
    validation = pd.DataFrame(validation)
    test = pd.DataFrame(test)

    # Ensure label column is INT (important)
    train[0] = train[0].astype(int)
    validation[0] = validation[0].astype(int)
    test[0] = test[0].astype(int)

    # =============================
    # 9. Save outputs
    # =============================
    os.makedirs(f"{base_dir}/train", exist_ok=True)
    os.makedirs(f"{base_dir}/validation", exist_ok=True)
    os.makedirs(f"{base_dir}/test", exist_ok=True)

    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

    print("✅ Processing complete. Files saved.")
