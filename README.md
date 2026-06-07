# 🚀 End-to-End MLOps Pipeline on AWS SageMaker

This project demonstrates a complete **MLOps pipeline** built using **Amazon SageMaker Pipelines**. It covers the full lifecycle of a machine learning model — from data preprocessing to training, evaluation, and deployment.

---

## 📌 Project Overview

This pipeline automates:

* 📥 Data ingestion from Amazon S3
* 🔄 Data preprocessing using `SKLearnProcessor`
* 🤖 Model training (XGBoost / Scikit-learn)
* 📊 Model evaluation (AUC, predictions)
* 📦 Model packaging & registration
* 🚀 Deployment via SageMaker endpoint

---

## 🏗️ Architecture

```
S3 (Raw Data)
     ↓
Processing Step (Preprocessing)
     ↓
Training Step (Model Training)
     ↓
Evaluation Step (Metrics)
     ↓
Model Registration
     ↓
Deployment (Endpoint / Batch Transform)
```

---

## ⚙️ Tech Stack

* **AWS SageMaker**
* **SageMaker Pipelines**
* **Python 3.10+**
* **Scikit-learn**
* **XGBoost**
* **Pandas / NumPy**
* **Amazon S3**
* **CloudWatch Logs**

---

## 📂 Project Structure

```
.
├── pipeline.py                  # SageMaker pipeline definition
├── preprocess-bank.py          # Data preprocessing script
├── train.py                    # Model training script
├── evaluate.py                 # Evaluation script
├── inference.py                # Inference logic for deployment
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* Bank Marketing Dataset (`bank-full.csv`)
* Stored in S3 bucket:

```
s3://<your-bucket>/data/bank-full.csv
```

---

## 🔄 Pipeline Steps

### 1. 🧹 Processing Step

* Reads raw data from S3
* Splits into:

  * Train
  * Validation
  * Test
* Saves processed data back to S3

### 2. 🧠 Training Step

* Trains XGBoost model
* Uses processed training data
* Outputs model artifact (`model.tar.gz`)

### 3. 📈 Evaluation Step

* Loads model + test data
* Generates predictions
* Computes AUC score

### 4. 📦 Model Registration

* Registers model in SageMaker Model Registry

### 5. 🚀 Deployment

* Deploys model as endpoint or batch transform

---

## ▶️ How to Run

### 1. Setup AWS

* Configure IAM Role with:

  * S3 access
  * SageMaker permissions

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Define Variables

```
bucket = "<your-bucket>"
region = "us-east-1"
```

---

### 4. Run Pipeline

```
pipeline.upsert(role_arn=role)
pipeline.start()
```

---

## ⚠️ Common Issues & Fixes

### ❌ File not found

```
No such file or directory: /opt/ml/processing/input/...
```

✅ Fix:

* Ensure correct S3 path
* Use dynamic file loading:

```
files = os.listdir("/opt/ml/processing/input")
```

---

### ❌ Feature mismatch (XGBoost)

```
ValueError: training data did not have the following fields
```

✅ Fix:

* Do NOT use `.values`
* Use DataFrame directly:

```
X_test = xgb.DMatrix(df)
```

---

### ❌ Instance type errors

* `ml.t3.medium` ❌ not allowed for Transform
* Use:

  * `ml.m5.large` ✅

---

### ❌ Quota issues

```
service limit ... is 0 Instances
```

✅ Fix:

* Request quota increase OR
* Use smaller instance types

---

## 💡 Best Practices

* Use dynamic file loading instead of hardcoding filenames
* Ensure consistent feature names across train/test
* Use smaller instance types for cost optimization
* Monitor logs in CloudWatch
* Version your pipeline

---

## 📈 Future Improvements

* ✅ Add CI/CD (CodePipeline / GitHub Actions)
* ✅ Automate retraining
* ✅ Add data validation (Great Expectations)
* ✅ Model monitoring (SageMaker Model Monitor)
* ✅ A/B testing

---

## 🙌 Conclusion

This project provides a **real-world MLOps pipeline** using AWS SageMaker, covering:

* Automation
* Scalability
* Production readiness

---

## ⭐ If you found this helpful

Give it a ⭐ on GitHub and share!

---
