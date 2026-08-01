Zenodo URL: https://doi.org/10.5281/zenodo.21737432
AgroCyberShield
A Reproducible FarmSecureNet Implementation for Cross-Domain IoT Intrusion Detection in Smart Farming Networks
AgroCyberShield is an explainable and edge-deployable cybersecurity framework designed to detect attacks affecting IoT-driven smart farming environments.
Its core detection model, FarmSecureNet, combines:
One-dimensional convolutional neural networks
Bidirectional long short-term memory
Temporal attention
These components learn local feature interactions, temporal attack behaviour, and security-relevant time steps.
The repository provides the complete implementation of the:
Data-processing pipeline
Feature-alignment pipeline
Model-training pipeline
Evaluation pipeline
Explainability pipeline
Ablation analysis
Cross-dataset testing
Edge-deployment pipeline
---
Overview
Smart farming infrastructures rely on connected:
Sensors
Gateways
Irrigation controllers
Weather stations
Cameras
Drones
Cloud services
Although these technologies support automated agricultural operations, they also introduce attack surfaces associated with:
Denial-of-service attacks
Malicious data injection
Botnets
Malware
Reconnaissance
Spoofing
Unauthorised access
AgroCyberShield addresses these risks through:
Dual-source learning from telemetry and network-flow data
Deterministic attack-label harmonisation
Leakage-safe chronological data partitioning
Shared behavioural feature-space alignment
Temporal sequence modelling
Hybrid CNN–BiLSTM–attention classification
SHAP-based and attention-based explanations
Cross-dataset generalisation analysis
Ablation and statistical evaluation
TorchScript and ONNX edge deployment
---
FarmSecureNet Architecture
FarmSecureNet processes a multivariate temporal input sequence using the following stages:
```text
Input Sequence
      │
      ▼
1D Convolutional Feature Extraction
      │
      ▼
ReLU, Batch Normalisation and Max Pooling
      │
      ▼
Bidirectional LSTM
      │
      ▼
Temporal Attention
      │
      ▼
Fully Connected Layers
      │
      ▼
Five-Class Threat Prediction
```
The model predicts one of the following unified classes:
Normal
DoS
Injection
Botnet
Malware
---
Main Features
Data Processing
TON_IoT telemetry, logs, and network-traffic ingestion
IoT-23 flow-level data ingestion
Optional CIC-IDS2017 external-generalisation adapter
Duplicate and invalid-row removal
Identifier and leakage-prone feature removal
Numerical and categorical missing-value imputation
Categorical feature encoding
Training-derived outlier clipping
Z-score standardisation
Zero-variance feature removal
Correlation-based redundancy filtering
Deterministic label mapping
Serialisation of fitted preprocessing objects
Cross-Domain Feature Alignment
Common behavioural feature schema
Packet, byte, duration, rate, timing, directionality, protocol, and temporal descriptors
Telemetry rolling statistics and lagged features
Flow-level behavioural signatures
Missing-feature indicators
Training-only shared PCA transformation
Configurable explained-variance threshold
Temporal Learning
Chronological 80:10:10 partitioning
Sequence length of 20 records
Stride of five records
Final-record sequence labelling
Device-, session-, and scenario-aware grouping
Prevention of windows crossing partition boundaries
Weighted sequence sampling
Optional flattened-sequence SMOTE reproduction mode
Evaluation
Accuracy
Macro and weighted precision
Macro and weighted recall
Macro and weighted F1-score
Class-wise performance
Multiclass ROC-AUC
Confusion matrices
Calibration analysis
Inference latency
Model size
Parameter count
Memory utilisation
Statistical comparison across multiple seeds
Explainability
SHAP global feature importance
SHAP class-wise explanations
Local prediction explanations
Temporal attribution heatmaps
Learned attention-weight visualisation
Optional LIME explanations
Attention-attribution consistency analysis
Deployment
PyTorch checkpoints
TorchScript export
ONNX export
Optional dynamic quantisation
Batch-size-one inference
Raspberry Pi and Jetson-compatible benchmark scripts
Configurable alert-generation interface
---
Repository Structure
```text
AgroCyberShield/
│
├── configs/
│   ├── base.yaml
│   ├── datasets.yaml
│   ├── farmsecurenet.yaml
│   ├── baselines.yaml
│   ├── ablation.yaml
│   └── edge.yaml
│
├── data/
│   ├── raw/
│   │   ├── ton_iot/
│   │   ├── iot23/
│   │   └── cicids2017/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── src/
│   └── agrocybershield/
│       ├── data/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── explainability/
│       ├── deployment/
│       └── utils/
│
├── scripts/
│   ├── prepare_ton_iot.py
│   ├── prepare_iot23.py
│   ├── prepare_cicids2017.py
│   ├── build_aligned_dataset.py
│   ├── train_farmsecurenet.py
│   ├── train_baselines.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   ├── run_cross_dataset.py
│   ├── run_explainability.py
│   ├── export_edge_model.py
│   ├── benchmark_edge.py
│   └── run_demo.py
│
├── notebooks/
├── tests/
├── outputs/
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```
---
Installation
1. Clone the Repository
```bash
git clone https://github.com/USERNAME/AgroCyberShield.git
cd AgroCyberShield
```
Replace `USERNAME` with the GitHub account or organisation name hosting the repository.
2. Create a Virtual Environment
Using Conda
```bash
conda create -n agrocybershield python=3.10 -y
conda activate agrocybershield
```
Using venv
```bash
python -m venv .venv
```
For Windows:
```bash
.venv\Scripts\activate
```
For Linux or macOS:
```bash
source .venv/bin/activate
```
3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
---
Quick Functional Test
The repository includes a synthetic-data demonstration that verifies:
Preprocessing
Sequence generation
Model training
Evaluation
Attention visualisation
Model export
Inference benchmarking
The full public datasets are not required for this test.
Run:
```bash
python scripts/run_demo.py --epochs 2
```
Run the unit tests:
```bash
pytest -q
```
> **Note:** The synthetic demonstration is intended only to validate software functionality. Its generated results must not be interpreted as research-paper performance.
---
Datasets
The original datasets are not redistributed with this repository.
Users must download them from their respective official sources and comply with their licences and usage conditions.
TON_IoT
TON_IoT contains heterogeneous IoT telemetry, operating-system logs, and network traffic.
It supports modelling:
Device-level anomalies
Telemetry manipulation
Local network attacks
Denial-of-service behaviour
Injection
Reconnaissance
Backdoor activity
Password attacks
Place the required files under:
```text
data/raw/ton_iot/
```
Aposemat IoT-23
IoT-23 contains benign and malware-generated IoT network traffic, including behaviour related to:
Mirai
Gafgyt
Torii
Scanning
Command-and-control communication
Other malicious activities
Place flow-level CSV or converted network-flow files under:
```text
data/raw/iot23/
```
Raw packet captures may be converted externally using CICFlowMeter or another compatible flow extractor.
The repository does not bundle CICFlowMeter.
CIC-IDS2017
CIC-IDS2017 is used only for controlled external-generalisation or supplementary evaluation.
It is not part of the principal TON_IoT–IoT-23 joint training workflow.
Place the files under:
```text
data/raw/cicids2017/
```
---
Data Preparation
Prepare TON_IoT
```bash
python scripts/prepare_ton_iot.py \
    --config configs/datasets.yaml
```
Prepare IoT-23
```bash
python scripts/prepare_iot23.py \
    --config configs/datasets.yaml
```
Prepare CIC-IDS2017
```bash
python scripts/prepare_cicids2017.py \
    --config configs/datasets.yaml
```
Build the Harmonised Dataset
```bash
python scripts/build_aligned_dataset.py \
    --config configs/base.yaml
```
The preprocessing workflow performs:
```text
Chronological ordering
        ↓
Cleaning and validation
        ↓
Missing-value imputation
        ↓
Categorical encoding
        ↓
Identifier removal
        ↓
Outlier clipping
        ↓
Training-fitted standardisation
        ↓
Zero-variance filtering
        ↓
Correlation filtering
        ↓
Behavioural feature alignment
        ↓
PCA dimensionality reduction
        ↓
Attack-label harmonisation
        ↓
Temporal sequence construction
        ↓
Training-only class balancing
```
All fitted preprocessing objects are stored and reused unchanged for validation, testing, explainability, and edge inference.
---
Attack-Label Harmonisation
The source-specific attack labels are mapped into a common operational taxonomy.
TON_IoT Mapping
Source Label	Unified Class
Normal or benign	Normal
DoS or DDoS	DoS
Injection or XSS	Injection
Backdoor or ransomware	Malware
Scanning, password, brute-force, or MitM	Malware
IoT-23 Mapping
Source Label	Unified Class
Benign	Normal
Mirai, Gafgyt, or Torii	Botnet
Explicit denial-of-service activity	DoS
Command-and-control, scanning, malicious download, or protocol abuse	Malware
No explicit injection equivalent	Not assigned to Injection
The original labels are retained in the processed metadata for traceability.
---
Training FarmSecureNet
Run:
```bash
python scripts/train_farmsecurenet.py \
    --config configs/farmsecurenet.yaml
```
The default configuration uses:
```yaml
epochs: 100
batch_size: 64
learning_rate: 0.001
weight_decay: 1.0e-5
early_stopping_patience: 10
sequence_length: 20
sequence_stride: 5
dropout: 0.30
lstm_hidden_size: 128
attention_dimension: 128
random_seed: 42
```
The best checkpoint is selected according to validation macro-F1 and stored under:
```text
outputs/checkpoints/
```
---
Training Baseline Models
Run:
```bash
python scripts/train_baselines.py \
    --config configs/baselines.yaml
```
Available baselines include:
One-dimensional CNN
BiLSTM
CNN–LSTM
GRU with attention
Transformer-based classifier
Classical tabular baselines where applicable
All baseline models use the same:
Preprocessing
Temporal partitions
Sequence windows
Label taxonomy
Evaluation protocol
---
Evaluation
Evaluate a trained checkpoint using:
```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/farmsecurenet_best.pt \
    --config configs/farmsecurenet.yaml
```
Generated outputs include:
```text
outputs/metrics/main_results.csv
outputs/metrics/classwise_results.csv
outputs/figures/confusion_matrix.png
outputs/figures/roc_curves.png
outputs/figures/classwise_metrics.png
outputs/figures/training_curves.png
```
---
Cross-Dataset Evaluation
Run:
```bash
python scripts/run_cross_dataset.py \
    --config configs/base.yaml
```
The supported protocols include:
Training on the aligned TON_IoT and IoT-23 training sets and testing on each domain separately
Training and testing within TON_IoT
Training and testing within IoT-23
Training on TON_IoT and testing on IoT-23
Training on IoT-23 and testing on TON_IoT
Optional CIC-IDS2017 external-generalisation testing
Because IoT-23 has no reliable Injection equivalent, cross-dataset evaluation can additionally report:
Shared four-class performance
Binary normal-versus-attack performance
---
Ablation Analysis
Run:
```bash
python scripts/run_ablation.py \
    --config configs/ablation.yaml
```
The implemented variants include:
Full FarmSecureNet
Without attention
Without BiLSTM
Without CNN
Without PCA
Without temporal feature engineering
Without class balancing
Without dual-source training
TON_IoT-only training
IoT-23-only training
Results are stored in:
```text
outputs/metrics/ablation_results.csv
outputs/figures/ablation_plot.png
```
---
Explainability
Run:
```bash
python scripts/run_explainability.py \
    --checkpoint outputs/checkpoints/farmsecurenet_best.pt \
    --config configs/farmsecurenet.yaml
```
The explainability module supports:
Global SHAP importance
Class-wise SHAP importance
Local feature attribution
Temporal attribution heatmaps
Attention-weight plots
Optional LIME explanations
Outputs are written to:
```text
outputs/explanations/
outputs/figures/shap_summary.png
outputs/figures/shap_classwise.png
outputs/figures/attention_heatmap.png
```
> **Important:** SHAP values and learned attention weights represent different explanatory signals. Attention visualisation should not be treated as a substitute for formal feature-attribution analysis.
---
Model Export
TorchScript and ONNX
```bash
python scripts/export_edge_model.py \
    --checkpoint outputs/checkpoints/farmsecurenet_best.pt \
    --config configs/edge.yaml
```
The exported models are stored under:
```text
outputs/edge_models/farmsecurenet.ts
outputs/edge_models/farmsecurenet.onnx
outputs/edge_models/farmsecurenet_int8.onnx
```
ONNX export requires the optional ONNX dependencies listed in `requirements.txt`.
---
Edge Benchmarking
Run:
```bash
python scripts/benchmark_edge.py \
    --model outputs/edge_models/farmsecurenet.onnx \
    --config configs/edge.yaml
```
The benchmark reports:
Mean inference latency
Median inference latency
Latency standard deviation
95th-percentile latency
Throughput
Model file size
Memory usage
CPU utilisation where supported
For meaningful deployment results, execute the benchmark directly on the target:
Raspberry Pi
Jetson Nano
Other edge platform
---
Output Structure
```text
outputs/
├── artifacts/
│   ├── scaler.joblib
│   ├── encoder.joblib
│   ├── pca.joblib
│   ├── feature_schema.json
│   ├── correlation_mask.json
│   └── label_mapping.json
│
├── checkpoints/
│   ├── farmsecurenet_best.pt
│   └── farmsecurenet_last.pt
│
├── edge_models/
│   ├── farmsecurenet.ts
│   ├── farmsecurenet.onnx
│   └── farmsecurenet_int8.onnx
│
├── explanations/
├── figures/
├── logs/
└── metrics/
```
---
Reproducibility
The implementation supports reproducibility through:
Fixed random seeds
Deterministic label mappings
Saved preprocessing parameters
Saved feature-selection masks
Saved PCA components
Versioned YAML configurations
Training and validation logs
Best and final checkpoints
Per-run predictions
Multiple-seed evaluation
Unit tests for data leakage and tensor shapes
Recommended experiment seeds are:
```text
42, 52, 62, 72, 82
```
Hardware, operating system, package versions, configuration files, and random seeds should be reported when publishing reproduced or extended results.
---
Testing
Run all tests:
```bash
pytest -q
```
The test suite covers:
Attack-label mapping
Chronological splitting
Sequence generation
Prevention of cross-partition windows
Preprocessing transformations
Model input and output shapes
Inference execution
Leakage checks
---
Limitations
The repository does not redistribute the original datasets.
IoT-23 is not an agriculture-specific dataset. It represents transferable network-layer threats applicable to agricultural IoT devices and gateways.
Cross-dataset feature alignment depends on semantically comparable behavioural descriptors.
Edge latency varies according to hardware, runtime, operating system, and model format.
SHAP explanations may be computationally expensive for large test sets.
Synthetic demonstration results verify execution only and do not represent real-world intrusion-detection performance.
---
Citation
When using this repository, cite the associated paper and the archived software release.
Research Paper
```bibtex
@article{agrocybershield,
  title   = {AgroCyberShield: Cybersecurity Strategies for Protecting IoT-Driven Smart Farming Networks},
  author  = {Raju, B. and Obulesu, Ooruchintala and Ramesh Babu, N. and Mandu, Swapna and Rao, Valluri Venkata Gopala and Saritha, B. Sri Lakshmi},
  journal = {To be updated},
  year    = {2026}
}
```
Software Citation
After the Zenodo release is created, add the software citation:
```bibtex
@software{agrocybershield_code,
  title     = {AgroCyberShield: Reproducible FarmSecureNet Implementation},
  author    = {Raju, B. and collaborators},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21737432},
  url       = {https://doi.org/10.5281/zenodo.21737432}
}
```
---
Code Availability
The implementation of AgroCyberShield and FarmSecureNet is available through this GitHub repository.
GitHub Repository:
```text
https://github.com/USERNAME/AgroCyberShield](https://github.com/bonagiriraju/AgroCyberShield)
```
Archived Release:
```text
Zenodo DOI: 10.5281/zenodo.21737432
Zenodo URL: https://doi.org/10.5281/zenodo.21737432
```
Replace these placeholders after publishing the repository and creating the Zenodo archive.
---
Licence
This project is released under the licence provided in the `LICENSE` file.
The datasets used by the project remain subject to their original licences and terms of use.
---
Contributing
Contributions that improve reproducibility, dataset compatibility, edge deployment, testing, or documentation are welcome.
A suggested contribution workflow is:
```bash
git checkout -b feature-name
git commit -m "Describe the implemented change"
git push origin feature-name
```
Then submit a pull request describing:
The implemented change
Its motivation
Affected modules
Validation or testing performed
---
Contact
For questions about the research or implementation, contact the corresponding author:
B. Raju
Email: raju.nestham@gmail.com
---
Disclaimer
AgroCyberShield is a research implementation.
It should not be used as the sole security mechanism for operational agricultural infrastructure without:
Independent validation
Secure deployment review
Continuous monitoring
Appropriate human oversight
