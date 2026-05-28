RealEstateVision is a production-oriented machine learning system for real-estate image operations. It simulates how a property technology company could validate image data, version datasets, benchmark multiple candidate models, and expose decision-ready results through an API and dashboard.

Along with designing the training exercise for candidate models model-training exercise, the project focuses more on the broader applied ML workflow such as data quality control, repeatable training, experiment tracking, model comparison across operational constraints, and clear deployment recommendations.

This project tries to simulate and address the real world business problem that is common in real-estate tech platforms: large volumes of property images need to be processed consistently, and often the best model is not necessarily the one with the highest accuracy score. In production, companies might look to balance prediction quality with serving speed, infrastructure cost, operational simplicity, and maintainability.

RealEstateVision addresses that problem by combining:
- image data validation and dataset quality checks
- dataset version awareness
- model training and experiment tracking
- model-level comparison across business-relevant criteria
- recommendation logic for deployment decisions
- API endpoints and dashboard views that make the system easier to inspect and use


## Business scenario being simulated

From business perspective, poor data quality can quietly degrade model performance, and raw model metrics alone are not enough to make a deployment decision. A production team needs to know:
- whether the dataset is trustworthy
- which dataset version was used
- which model is active in serving
- how candidate models compare on quality, latency, cost, and maintainability
- which model should actually be deployed and why


## API and dashboard sections

| API script | API section | What it does | Why it matters |
|---|---|---|---|
| `predict.py` | `/predict` | Runs inference on a room image using the currently active model or optional request-level overrides. | Demonstrates the serving layer of the system and shows how trained models are exposed for real prediction use. |
| `models.py` | `/models` | Lists available model artifacts, version tags, and the currently active serving model. | Helps users inspect which models are ready for deployment and which model is currently being used by the API. |
| `data_validation.py` | `/data_validation` | Summarizes dataset validation results, including image quality checks and data reliability signals. | Supports trust in the training and evaluation pipeline by showing whether the underlying image data passed validation standards. |
| `dashboard.py` | `/dashboard` | Provides a decision-ready overview of candidate models, including benchmark comparisons across quality, latency, cost, and maintainability, along with a final recommendation. | Translates experiment results into a deployment recommendation and highlights the trade-offs that matter in a production ML setting. |
| `runs.py` | `/runs` | Lists tracked experiment runs with associated metrics, metadata, and run identifiers. | Preserves experiment traceability and helps inspect how individual training runs were configured and evaluated. |
| `main.py` | `/health` | Reports the operational health of the RealEstateVision API, including database connectivity, model availability, and system readiness. | Makes it easier to verify that the API is running correctly and that key dependencies are available. |


## System architecture

RealEstateVision is organized as a production-style ML pipeline rather than a notebook-first workflow. The system moves from raw image intake to validation, dataset versioning, candidate model training, experiment tracking, evaluation, recommendation, and finally API-based serving.

```text
Raw Images
   ↓
Data Ingestion and Metadata Preparation   (PySpark, Python, SQLite)
   ↓
Data Validation and Cleaning          (CleanVision, Python, SQLite)
   ↓
Validated / Cleaned Dataset           (SQLite, Pandas, PIL)
   ↓
Dataset Version Registration          (DVC, SQLite)
   ↓
Training Pipeline                     (PyTorch, Torchvision)
   ↓
Experiment Tracking                   (MLflow)
   ↓
Evaluation Reports                    (scikit-learn, JSON, CSV)
   ↓
Model Comparison and Recommendation   (Python scoring logic, FastAPI dashboard router)
   ↓
API Serving Layer                     (FastAPI, Pydantic)
   ↓
Dashboard and Decision Views          (FastAPI endpoints, MLflow metadata, comparison tables)
```

Each stage has a clear role: validation protects the training set, versioning makes experiments reproducible, training produces candidate models, MLflow keeps runs traceable, evaluation turns run outputs into model-level comparisons, and the API plus dashboard expose both operational status and deployment decisions.


## Data quality and traceability

Data validation is treated as part of the ML system, not a side task. The validation layer catches corrupt, blurry, dark, bright, low-information, and duplicate images by leveraging CleanVision package, so training only uses trustworthy data.

MLflow tracks each run with its model name, dataset version, metrics, and artifacts, which makes the system auditable and easy to compare across versions.


## Training and evaluation


The core task is room-type classification across five classes: bathroom, bedroom, dining room, kitchen, and living room. The candidate models — MobileNetV3-Small, EfficientNet-B0, and ResNet-18 — were chosen to show the trade-off between lightweight and heavier image backbones under realistic constraints.

Evaluation goes beyond accuracy. The pipeline also measures latency, throughput, cost per 1,000 images, maintainability, and parameter count, which makes the comparison closer to a real deployment decision than a pure benchmark.


## Dashboard purpose

The dashboard is not just an experiment log viewer. It is organized to answer three questions: what is happening now, which models are competing, and which model should be deployed.

- **Summary** shows the current state of the system and the leading candidates in each evaluation metric.
- **Models Compare** ranks models across quality, latency, cost, and maintainability.
- **Recommendation** converts those results into a clear deployment choice.


## Repo Structure

```text
RealEstateVision/
├── api/                     # FastAPI application and route handlers
│   ├── main.py
│   ├── dependencies.py
│   ├── inference.py
│   └── routers/
│       ├── predict.py
│       ├── models.py
│       ├── data_validation.py
│       ├── dashboard.py
│       └── runs.py
├── src/
│   ├── ingestion/              # dataset extraction and subset preparation
│   │   └── prepare_mit_subset.py
│   ├── metadata/               # PySpark-based metadata processing
│   │   └── spark_process_metadata.py
│   ├── database/               # load processed metadata into SQLite
│   │   └── load_metadata_to_sqlite.py
│   ├── validation/             # image quality and validation pipeline
│   │   └── run_image_quality_validation_final.py
│   └── training/               # model training and evaluation
│       └── train_classifier.py
|── data_versioning/                # record dataset version in SQL
|   |── record_dataset_version.py
├── data/
│   ├── raw/                # source dataset
│   │   └── mit_indoor_subset/
│   ├── analytics/              # quality and metadata summaries
│   │   └── quality/
│   └── processed/
|       └── metadata/               # full metadata extracted table
|       └── validation_reports/             # data validation reports created using CleanVision package
|       └── cleaned/                # resulting cleaned data
│       └── evaluation_reports/             # model reports and comparison tables
├── checkpoints/                # best-checkpoint snapshots during training
├── models/             # exported model weights for serving
├── db/             # SQLite database and dataset metadata
├── mlruns/             # MLflow tracking artifacts
├── Dockerfile
├── docker-compose.yml              # local multi-service orchestration
├── requirements.txt                # core project dependencies
├── requirements.api.txt                # API-specific dependencies
├── README.md
└── .dvc/               # DVC configuration
```












