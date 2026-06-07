````markdown
# ⚖️ DebateIQ

DebateIQ is an NLP-based argument mining system that identifies **claims**, **premises**, and **stance (PRO / CON / NONE)** from argumentative text.

The project combines **Sentence-BERT (SBERT)** embeddings with a **BiLSTM-based multi-task classifier** and provides predictions through a **FastAPI backend** and an interactive **Streamlit dashboard**.

---

## Features

- Claim Detection
- Premise Detection
- Stance Classification (PRO / CON / NONE)
- Interactive Argument Tree Visualization
- Confidence-Based Predictions
- FastAPI REST API
- Streamlit Dashboard
- ONNX Export Support
- Dockerized Deployment

---

## Architecture

```text
Input Text
    ↓
Sentence Splitting
    ↓
SBERT Embeddings
    ↓
BiLSTM Context Encoder
    ↓
Multi-Task Prediction Heads
    ├── Claim Detection
    ├── Premise Detection
    └── Stance Classification
    ↓
FastAPI Backend
    ↓
Streamlit Dashboard
````

---

## Tech Stack

### Machine Learning

* PyTorch
* Sentence Transformers (SBERT)
* Scikit-learn
* NumPy
* ONNX Runtime

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit
* Plotly

### Visualization

* NetworkX
* PyVis
* Matplotlib

### Deployment

* Docker
* Docker Compose

---

## Project Structure

```text
debateiq/
├── configs/
│   └── config.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── train/
│   └── evaluate/
├── api/
│   └── main.py
├── streamlit_app/
│   └── app.py
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── run.py
```

---

## Installation

```bash
git clone https://github.com/ridhisharmaa/debateIQ-project.git

cd debateIQ-project/debateiq

pip install -r requirements.txt

pip install -e .
```

---

## Running the Project

### Exploratory Data Analysis

```bash
python run.py eda
```

### Model Training

```bash
python run.py train
```

### Model Evaluation

```bash
python run.py evaluate
```

### FastAPI Backend

```bash
python run.py api
```

API Documentation:

```text
http://localhost:8000/docs
```

### Streamlit Dashboard

```bash
python run.py streamlit
```

Dashboard:

```text
http://localhost:8501
```

---

## Docker

```bash
docker-compose up --build
```

Services:

* FastAPI → http://localhost:8000
* Streamlit → http://localhost:8501

---

## Example Input

**Topic**

```text
Education
```

**Text**

```text
Online education is better than traditional classrooms.
It saves travel time and allows students to learn at their own pace.
Therefore schools should invest more in online learning platforms.
```

The system analyzes each sentence and predicts:

* Claim / Non-Claim
* Premise / Non-Premise
* Stance (PRO / CON / NONE)

---

## Future Improvements

* Fine-tuned Transformer Models
* Larger Debate Datasets
* Argument Relation Extraction
* Real-Time Collaboration Features
* Hugging Face Spaces Deployment

---


