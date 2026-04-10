# Iris Flower Classification — Production-Ready ML Project

## 📌 Overview

This project implements an end-to-end Machine Learning pipeline to classify iris flowers into three species based on their physical measurements.

This implementation focuses on **production-level practices**, including modular pipelines, experiment tracking, API deployment, and containerization.

---

## Problem

Given four features:

* Sepal length
* Sepal width
* Petal length
* Petal width

Predict the correct iris species:

* Setosa
* Versicolor
* Virginica

---

## Approach

The project follows a complete ML lifecycle:

1. Data loading
2. Data preprocessing
3. Model training (Random Forest)
4. Evaluation
5. Model persistence
6. Inference (script + API)


---

## Tech Stack

* Python
* Scikit-learn
* FastAPI
* MLflow
* Docker

---


## How to Run

### 1. Clone repository

```
git clone <your-repo-url>
cd iris-classification
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train model

```
python train.py
```

---

## Inference (CLI)

```
python predict.py
```

---

## 🌐 API Usage

Start server:

```
uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🧠 MLflow Tracking

Start MLflow UI:

```
mlflow ui
```

Track:

* Parameters
* Metrics
* Model versions

---

## 🐳 Docker

Build:

```
docker build -t iris-api .
```

Run:

```
docker run -p 8000:8000 iris-api
```

---

## 🧪 Tests

```
python -m pytest tests/
```

---

## 💡 Key Highlights

* Modular ML pipeline
* Config-driven training
* Experiment tracking with MLflow
* REST API for inference
* Dockerized deployment

---

## 👤 Author

Arthur Lincoln da Paz Cristovao
