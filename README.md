# 🏡 RealEstateVision

Production-Grade ML Pipeline for Property Image Classification, Metadata Engineering, Data Quality, and Build-vs-Buy Evaluation

***

## 📌 Overview

RealEstateVision is an end-to-end, production-oriented machine learning project designed to simulate how modern real-estate technology companies process, validate, benchmark, and serve property image intelligence workflows at small-to-medium scale.[1]

The project focuses less on chasing the highest possible model accuracy and more on building a reliable system around data ingestion, metadata engineering, data quality monitoring, reproducible experimentation, and evidence-based build-vs-buy decisions under real-world constraints such as latency, cost, and maintainability.[1]

For the current implementation scope, the project is intentionally narrowed to **indoor room-type image classification** using the MIT Indoor Scenes dataset, with segmentation removed from the MVP so the system can be executed credibly within a short timeline on a MacBook M2.[2][1]

***

## 🎯 Objectives

- Build a robust pipeline for ingesting and validating indoor property-style images.[1]
- Engineer structured image metadata for downstream analytics and quality monitoring.[1]
- Use PySpark as a metadata engineering and analytics layer, not as the image-processing engine.[1]
- Integrate SQL as a relational metadata and experiment registry layer.[1]
- Train and evaluate lightweight room-classification models using PyTorch on Apple Silicon-friendly hardware.[1]
- Develop a structured evaluation framework comparing open-source, simulated API, and custom-trained approaches across quality, latency, cost, and maintainability.[1]
- Simulate production ML infrastructure with FastAPI, Docker, local storage, and versioned datasets.[1]
- Deliver a clear build-vs-buy recommendation based on quantitative and operational evidence.[1]

***

## 🧠 Key Insight

In production ML systems, the hardest problems are often not modeling itself, but data reliability, metadata traceability, validation rigor, and system-level decision-making under constraints.[1]

RealEstateVision is designed to demonstrate those production-oriented skills in a compact but realistic portfolio project aligned with junior data scientist / data engineer responsibilities.[1]

***

## 🏗️ Current Architecture

```text
Raw Images
   ↓
Metadata Extraction (Python / OpenCV / PIL)
   ↓
PySpark Processing Layer
   ↓
SQL Metadata Store
   ↓
Quality Analytics
   ↓
Data Validation
   ↓
Data Versioning (DVC)
   ↓
ML Training Pipeline (PyTorch)
   ↓
Evaluation Engine
   ↓
Experiment Tracking (MLflow)
   ↓
FastAPI + Docker
```

### Core Components

- Data ingestion and raw image organization.[1]
- Metadata extraction using Python, PIL, and OpenCV.[1]
- Metadata processing and analytical summaries using PySpark in local mode.[1]
- Relational metadata storage using SQLite, with PostgreSQL as a natural future upgrade path.[1]
- Data quality validation for blur, brightness, contrast, duplicates, and corruption.[1]
- Dataset versioning with DVC.[1]
- Image classification with lightweight PyTorch models.[1]
- Structured evaluation and benchmarking engine.[1]
- Experiment tracking with MLflow.[1]
- REST API serving layer with FastAPI.[1]
- Dockerized local deployment.[1]

***

## ⚙️ Tech Stack

### Core ML and Data
- Python 3.10+[1]
- NumPy, Pandas, OpenCV, Pillow[1]
- PyTorch with MPS backend for Apple Silicon[1]
- Scikit-learn[1]

### Metadata Engineering and Analytics
- PySpark (local mode, metadata engineering and analytics only)
- PyArrow for Parquet outputs
- SQLite for relational metadata storage and queryable experiment summaries

### Pipelines and Versioning
- DVC for dataset versioning.[1]
- Prefect as an optional orchestration extension, not required for the MVP.[1]

### Experiment Tracking
- MLflow local tracking server.[1]

### Backend / Serving
- FastAPI[1]
- Uvicorn[1]

### Containerization
- Docker with Apple Silicon-compatible images.[1]

### Storage
- Local filesystem simulating an object-storage style layout.[1]
- SQL database for structured metadata and benchmark records.

***

## 💻 Hardware Constraints and Design Choices

This project is designed to run efficiently on:

- Apple Silicon (M1 / M2)[1]
- 16 GB RAM[1]
- No mandatory GPU dependency, with optional MPS acceleration for training.[1]

### Design Choices

- Lightweight models such as MobileNet or ResNet18 instead of large vision transformers.[1]
- Dataset subset kept intentionally small and relevant for room classification.[1]
- Batch inference instead of large-scale distributed image processing.[1]
- PySpark used only where it is a good fit: structured metadata engineering and analytics, not direct image transformation.[1]
- SQLite used for practical local execution, with production-oriented schema design.[1]

***

## 📂 Dataset

### Primary Dataset
- MIT Indoor Scenes dataset (67 indoor categories, 15,620 JPG images, official train/test split files).[2]

### Current Scope
The current MVP uses a real-estate-relevant subset of MIT Indoor categories for room classification, such as:

- bathroom
- bedroom
- dining_room
- kitchen
- livingroom

This subset is chosen to keep the project aligned with property imagery while remaining realistic to execute quickly and reproducibly.[2][1]

### Data Characteristics
- Indoor scenes relevant to real-estate use cases.[1]
- Variation in lighting, room layout, image quality, and composition.[1]
- Official benchmark split available for reproducible training and evaluation.[2]

### Storage Structure

```text
data/
  external/
    mit_indoor/
      indoorCVPR_09.tar
      TrainImages.txt
      TestImages.txt
      Images/
  raw/
    mit_indoor_subset/
      train/
      test/
  processed/
    metadata/
      extracted/
      spark/
    validation/
  analytics/
    quality/
  versions/
```

***

## 🧼 Data Pipeline

### Responsibilities
- Image ingestion from the official MIT split files.[2]
- Controlled subset selection for relevant room categories.
- Metadata extraction using Python / PIL / OpenCV.[1]
- Metadata processing and profiling using PySpark.
- Relational metadata loading into SQLite.
- Data validation and cleaned-dataset generation.[1]

### Metadata Extraction Fields
Examples of extracted fields include:

- source dataset
- split
- class name
- label id
- file path
- width and height
- number of channels
- file format and extension
- aspect ratio
- pixel count
- file size
- md5 hash
- ingestion timestamp

### Data Quality Checks
- Blur detection using Laplacian variance.[1]
- Brightness threshold checks.[1]
- Contrast threshold checks.[1]
- Duplicate detection using file hash and perceptual hash.[1]
- Corrupt file detection.[1]
- Basic metadata anomaly detection such as tiny images or odd aspect ratios.

### Outcome
A clean, versioned, queryable, and analytically profiled dataset ready for model training and evaluation.[1]

***

## 🗃️ SQL Integration

SQL is used as the structured metadata and experiment registry layer for the project.

### Initial Database Choice
- SQLite for local development and demonstration.
- PostgreSQL as the natural production upgrade path.

### Core Tables
- `images` — ingested image metadata.
- `quality_checks` — blur, brightness, contrast, duplicate, and validity results.
- `dataset_versions` — DVC-linked dataset snapshots.
- `training_runs` — compact summaries of training configurations and outcomes.
- `benchmark_results` — build-vs-buy comparison outputs.

This layer makes the pipeline queryable, supports reproducible analytics, and demonstrates practical SQL usage relevant to the target role.[1]

***

## 🔁 Data Versioning

Using DVC, datasets are:

- Version-controlled.[1]
- Reproducible.[1]
- Traceable across experiments.[1]

Each experiment should be linked to:

- Dataset version.[1]
- Model configuration.[1]
- Evaluation results.[1]
- SQL metadata records and MLflow runs.

***

## 🧠 Machine Learning Pipeline

### 1. Image Classification (Current MVP)
- Task: room type classification such as kitchen, bedroom, bathroom, dining room, and living room.[1]
- Models: MobileNet / ResNet18 / similarly lightweight backbones.[1]
- Training target: a reproducible baseline model with tracked metrics and inference timing.[1]

### 2. Image Enhancement (Optional / Secondary)
- Brightness and contrast correction using classical approaches first.[1]
- Kept as a secondary extension only if time allows.

### Removed from MVP
- Image segmentation is intentionally excluded from the current execution plan to keep the project realistic within the available timeline.

***

## ⚔️ Evaluation Framework

A structured benchmarking system compares:

1. Open-source baseline models (PyTorch).[1]
2. Simulated external API approach.[1]
3. Custom-trained model approach.[1]

### Metrics

#### Quality
- Accuracy.[1]
- F1-score.[1]
- Confusion matrix analysis.

#### Performance
- Inference latency (ms/image).[1]
- Throughput (images/sec).[1]

#### Cost (Simulated)
- Cost per 1,000 images.[1]
- CPU vs GPU estimate.[1]
- Managed API estimate versus in-house batch inference.

#### Maintainability
- Complexity score (qualitative).[1]
- Dependency footprint.[1]
- Deployment effort.[1]
- Operational simplicity.

### Example Evaluation Table

| Approach | Accuracy | Latency (ms/image) | Cost / 1k Images | Maintainability |
|---|---:|---:|---:|---|
| Open-source baseline | 0.89 | 120 | €0.50 | Medium |
| Simulated API | 0.91 | 80 | €4.20 | High |
| Custom-trained | 0.93 | 150 | €0.70 | Low |

These values are illustrative placeholders until the real benchmark pipeline is executed.[1]

***

## 📈 Experiment Tracking

Using MLflow:

- Track experiments.[1]
- Log metrics and parameters.[1]
- Compare runs.[1]
- Store model artifacts and confusion matrices.[1]
- Link runs to dataset versions and SQL metadata summaries.

***

## 🌐 API Layer

FastAPI service exposing:

- `/health` → system health check.[1]
- `/predict` → run room-type inference on an image.[1]
- `/evaluate` → optionally trigger or summarize benchmark outputs.[1]

### Supported Modes
- Single-image prediction.
- Batch processing simulation.[1]
- Real-time inference simulation.[1]

***

## 🐳 Docker Setup

The MVP Docker setup is intentionally lightweight and may include:

- API service container.[1]
- Optional MLflow service container.[1]
- Local volume mounts for data and artifacts.

### Goals
- Reproducibility.[1]
- Clean environment setup.[1]
- Deployment readiness for a local portfolio demo.[1]

***

## ☁️ Production Simulation

While running locally, the system simulates:

- S3-like local storage structure.[1]
- Batch job processing.[1]
- Queue-like inference behavior at a basic level.[1]
- CPU vs GPU trade-off analysis.[1]
- Queryable metadata and experiment summaries through SQL.

***

## 📊 Dashboard (Optional Extension)

A lightweight Streamlit dashboard may be added if time allows to:

- Visualize class distribution and data quality summaries.[1]
- Compare benchmark approaches.[1]
- Inspect sample predictions and confusion matrices.[1]

This is optional and not required for the MVP.

***

## 📌 Build vs Buy Analysis

### Key Question
Should the system:

- Build in-house models?[1]
- Use open-source models?[1]
- Rely on external APIs?[1]

### Current Focus
The first build-vs-buy analysis will focus on **room classification only**, not segmentation, to ensure the recommendation is based on a completed and defensible benchmark scope.[1]

### Expected Outcome
A likely recommendation for the MVP is a hybrid perspective:

- Open-source models are cost-efficient and flexible for batch classification.[1]
- Simulated APIs may offer lower operational burden and faster perceived deployment.[1]
- Custom models may be justified only if domain-specific accuracy improvements are meaningful enough to offset engineering overhead.[1]

***

## 🚀 Execution Plan

### Step 1 — Project Setup
- Repository structure
- Python environment
- Base dependencies
- Git and DVC initialization

### Step 2 — Data Acquisition and Metadata Engineering
- Download and extract MIT Indoor dataset.[2]
- Select real-estate-relevant class subset.
- Copy images into project raw structure using official split files.[2]
- Extract metadata in Python / OpenCV / PIL.
- Process metadata with PySpark.
- Load metadata into SQLite.

### Step 3 — Data Validation and Quality Reporting
- Run blur, brightness, contrast, duplicate, and corruption checks.[1]
- Produce validation reports and cleaned dataset.
- Persist quality results into SQL.

### Step 4 — Data Versioning
- Track cleaned dataset with DVC.[1]
- Record dataset versions in SQL.

### Step 5 — Baseline Model Training
- Train one lightweight room-classification model.[1]
- Save artifacts and evaluation outputs.

### Step 6 — Evaluation Engine
- Benchmark quality, latency, simulated cost, and maintainability.[1]
- Compare open-source, simulated API, and custom-trained approaches.[1]

### Step 7 — Experiment Tracking
- Log runs to MLflow.[1]
- Link MLflow runs with SQL and DVC references.

### Step 8 — API Layer
- Build minimal FastAPI endpoints.[1]

### Step 9 — Dockerization
- Package the API and supporting services with Docker.[1]

### Step 10 — Final Reporting / Optional Dashboard
- Summarize results.
- Optionally expose simple dashboard views.[1]

***

## ▶️ How to Run

```bash
# Clone repo
git clone https://github.com/yourusername/RealEstateVision.git
cd RealEstateVision

# Setup environment
conda create -n realestatevision python=3.10 -y
conda activate realestatevision

# Install dependencies
pip install -r requirements.txt

# Initialize SQL database
python src/utils/init_db.py

# Run metadata ingestion
python src/ingestion/prepare_mit_subset.py

# Run PySpark metadata processing
python src/metadata/spark_process_metadata.py

# Run validation
python src/validation/validate_images.py

# Track experiments
mlflow ui

# Run API
uvicorn api.main:app --reload
```

***

## 📌 MVP Scope

To keep the project realistic and executable within a short time frame, the MVP includes:

- Classification only, no segmentation.[1]
- MIT Indoor subset only.[2]
- Local execution only.[1]
- PySpark for metadata only.
- SQLite for structured metadata and results.
- Minimal FastAPI and Docker layers.

This scope is intentionally designed to maximize credibility, completeness, and alignment with junior DS/DE responsibilities rather than breadth for its own sake.[1]

***

## 📌 Future Improvements

- PostgreSQL instead of SQLite for multi-user workflows.
- Prefect orchestration for automated scheduled runs.[1]
- Real AWS integration with S3 and SageMaker.[1]
- GPU scaling experiments.[1]
- Active learning loop for annotation.[1]
- Dashboard polish and richer monitoring.[1]
- Image enhancement as a fully benchmarked second task.[1]

***

## 👤 Author

**Vasu**  
Data Scientist | ML Engineer  
Focused on building production-grade ML systems with strong data foundations.[1]

***

## 💬 Closing Note

This project is not about achieving the highest accuracy. It is about demonstrating how to build reliable, traceable, and decision-driven ML systems around image data in a way that reflects real-world R&D and applied ML engineering practice.[1]

Sources
[1] RealEstateVision_README.md https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_3778e485-9397-4d9a-9ba5-977a1d8b832f/ddb9397f-048b-4d65-a601-aae18a4239fe/RealEstateVision_README.md
[2] Indoor Scene Recognition, CVPR 09 https://web.mit.edu/torralba/www/indoor.html
🏡 RealEstateVision

Production-Grade ML Pipeline for Property Image Processing, Data Quality, and Build-vs-Buy Evaluation

⸻

📌 Overview

RealEstateVision is an end-to-end, production-oriented machine learning system designed to simulate how modern real-estate tech companies process and optimize property images at scale.

This project focuses not only on building ML models, but on engineering reliable data pipelines, ensuring data quality, and making informed build-vs-buy decisions under real-world constraints such as latency, cost, and maintainability.

It reflects how applied ML systems operate in production environments where data reliability and system design matter as much as model accuracy.

⸻

🎯 Objectives
    •    Build a robust data pipeline for ingesting and validating real estate images
    •    Design data quality monitoring systems to detect issues early
    •    Implement multiple ML approaches (open-source + API-based)
    •    Develop a structured evaluation framework comparing:
    •    Quality
    •    Latency
    •    Cost
    •    Maintainability
    •    Simulate production ML infrastructure using Docker and APIs
    •    Deliver a clear build-vs-buy recommendation system

⸻

🧠 Key Insight

In production ML systems, the hardest problems are not modeling —
they are data reliability, system design, and decision-making under constraints.

⸻

🏗️ System Architecture

Raw Images → Data Validation → Data Versioning → ML Pipeline → Evaluation Engine → Tracking → API + Dashboard

Components:
    •    Data ingestion & validation
    •    Data versioning (DVC)
    •    ML inference pipelines
    •    Evaluation & benchmarking engine
    •    Experiment tracking (MLflow)
    •    REST API for serving results
    •    Dockerized environment

⸻

⚙️ Tech Stack (Optimized for MacBook M2, 16GB RAM)

Core
    •    Python 3.10+
    •    NumPy, Pandas, OpenCV
    •    PyTorch (MPS backend for Apple Silicon)
    •    Scikit-learn

Pipelines & Orchestration
    •    Prefect (lightweight, Mac-friendly)
    •    DVC (data versioning)

Experiment Tracking
    •    MLflow (local tracking server)

Backend / Serving
    •    FastAPI
    •    Uvicorn

Containerization
    •    Docker (Apple Silicon compatible images)

Storage
    •    Local filesystem (simulating S3 structure)

⸻

💻 Hardware Constraints Consideration

This project is designed to run efficiently on:
    •    Apple Silicon (M1/M2)
    •    16 GB RAM
    •    No GPU dependency (MPS optional acceleration)

Design Choices:
    •    Lightweight models (MobileNet, small YOLO variants)
    •    Batch inference instead of large-scale parallelism
    •    Dataset size capped (~5–10k images)

⸻

📂 Dataset

Primary Sources:
    •    Airbnb property image datasets (Inside Airbnb)
    •    Kaggle real estate image datasets

Data Characteristics:
    •    Indoor scenes (bedroom, kitchen, living room)
    •    Varying lighting, quality, and composition

Storage Structure:

data/
  raw/
  processed/
  annotations/
  versions/
  
  🧼 Data Pipeline

Responsibilities:
    •    Image ingestion
    •    Format normalization (resolution, size)
    •    Metadata extraction
    •    Data validation

Data Quality Checks:
    •    Blur detection (Laplacian variance)
    •    Brightness/contrast thresholds
    •    Duplicate detection (perceptual hashing)
    •    Corrupt file detection

Outcome:

A clean, versioned, and reliable dataset ready for training and evaluation.

⸻

🔁 Data Versioning

Using DVC, datasets are:
    •    Version-controlled
    •    Reproducible
    •    Traceable across experiments

Each experiment is linked to:
    •    Dataset version
    •    Model configuration
    •    Evaluation results

⸻

🧠 Machine Learning Pipeline

Tasks:

1. Image Classification
    •    Room type classification (kitchen, bedroom, etc.)
    •    Model: MobileNet / ResNet (lightweight)

2. Image Segmentation
    •    Identify walls, furniture, floor regions
    •    Model: U-Net (light version)

3. Image Enhancement
    •    Brightness and contrast correction
    •    Classical + learned approaches

⸻

⚔️ Evaluation Framework (Core Contribution)

A structured benchmarking system comparing:

Approaches:
    1.    Open-source models (PyTorch)
    2.    Simulated external APIs
    3.    Custom-trained models

Metrics:

Quality:
    •    Accuracy
    •    F1-score
    •    IoU (segmentation)

Performance:
    •    Inference latency (ms/image)
    •    Throughput (images/sec)

Cost (Simulated):
    •    Cost per 1,000 images
    •    GPU vs CPU estimates

Maintainability:
    •    Complexity score (qualitative)
    •    Dependency footprint
    •    Deployment effort

⸻

📊 Example Evaluation Table

Approach    Accuracy    Latency (ms)    Cost/1k Images  Maintainability 

Open-source 0.89    120 €0.50   Medium
API (Simulated) 0.91    80  €4.20   High
Custom  0.93    150 €0.70   Low

📈 Experiment Tracking

Using MLflow:
    •    Track experiments
    •    Log metrics
    •    Compare runs
    •    Store artifacts

⸻

🌐 API Layer

FastAPI service exposing:
    •    /predict → Run inference
    •    /evaluate → Run benchmark
    •    /health → System status

Supports:
    •    Batch processing
    •    Real-time inference simulation

⸻

🐳 Docker Setup

Each component is containerized:
    •    API service
    •    ML pipeline
    •    MLflow tracking server

Ensures:
    •    Reproducibility
    •    Clean environment setup
    •    Deployment readiness

⸻

☁️ Production Simulation

While running locally, the system simulates:
    •    S3-like storage structure
    •    Batch job processing
    •    Queue-based inference (basic simulation)
    •    CPU vs GPU trade-offs

⸻

📊 Dashboard (Optional Extension)

Simple dashboard (Streamlit):
    •    Visualize model performance
    •    Compare approaches
    •    Inspect sample predictions

⸻

📌 Build vs Buy Analysis

Key Question:

Should we:
    •    Build in-house models?
    •    Use open-source?
    •    Rely on external APIs?

Findings (Example):
    •    Segmentation: Open-source models are cost-efficient at scale
    •    Classification: APIs offer better latency and lower maintenance
    •    Enhancement: Hybrid approach recommended

⸻

🧾 Final Recommendation

A hybrid architecture:
    •    Open-source for heavy tasks
    •    APIs for latency-critical components
    •    Custom models for domain-specific optimization

⸻

🚀 How to Run

# Clone repo
git clone https://github.com/yourusername/RealEstateVision.git
cd RealEstateVision

# Setup environment
conda create -n realestatevision python=3.10
conda activate realestatevision

# Install dependencies
pip install -r requirements.txt

# Run MLflow
mlflow ui

# Run API
uvicorn api.main:app --reload

📌 Future Improvements
    •    Kubernetes deployment
    •    Real AWS integration (S3, SageMaker)
    •    GPU scaling experiments
    •    Active learning loop for annotation

⸻

👤 Author

Vasu
Data Scientist | ML Engineer
Focused on building production-grade ML systems with strong data foundations

⸻

💬 Closing Note

This project is not about achieving the highest accuracy —
it is about demonstrating how to build reliable, scalable, and decision-driven ML systems in real-world environments.




