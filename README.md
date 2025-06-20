# SciX Multi-Label Classification

## Project Overview
A machine learning project for automated multi-label classification of scientific literature using the SciX dataset. The project implements and compares several approaches including TF-IDF based models and transformers.

## Project Structure
```
SciX-MultiLabel-Classification/
├── configs/                     # Configuration files
│   ├── base_config.yaml        # Model hyperparameters
│   └── preprocessing.yaml      # Text preprocessing settings
│
├── data/                       # Data directory
│   ├── raw/                   # Original SciX dataset
│   └── processed/             # Cleaned and processed data
│
├── fig/                       # Figure directory
│
├── img/                       # Image directory
│
├── models/                     # Model artifacts
│   ├── baseline/              # TF-IDF based models
│   └── transformer/           # BERT-based models
│
├── notebooks/                  # Analysis notebooks
│   ├── 01_data_exploration.ipynb    # Dataset analysis
│   ├── 02_preprocessing.ipynb       # Text preprocessing
│   ├── 03_model_training.ipynb      # Main training 
│   ├── 03.1_transformer_model.ipynb # BERT model training
│   ├── 04_results_analysis.ipynb    # Baseline analysis
│   └── 04.1_transformer_analysis.ipynb # BERT analysis
│
├── src/                       # Source code
│   ├── data_preprocessing.py  # Text cleaning utilities
│   ├── evaluate.py           # Evaluation metrics
│   ├── models.py             # Model implementations
│   ├── trains.py            # Training pipelines
│   └── utils.py             # Helper functions
│
├── .gitattributes
├── .gitignore
├── LICENCE
├── README.md
└── requirements.txt
```

## Models Implemented
1. **Baseline Models**
   - TF-IDF + Logistic Regression (Micro-F1: 0.23)
   - TF-IDF + Random Forest
   - TF-IDF + XGBoost

2. **Advanced Models**
   - DistilBERT Transformer (Micro-F1: 0.35)
   - Improved Multi-Label with threshold optimization

## Key Features
- Scientific text preprocessing optimized for academic papers
- Multi-label classification handling 1000+ possible labels
- Dynamic threshold optimization per class
- Detailed performance analysis and error diagnostics
- Comparison between traditional ML and transformer approaches

## Requirements

- **Python version:** 3.8 or higher
- All dependencies are listed in `requirements.txt`

## Data Download

The SciX dataset is automatically downloaded using the `datasets` library in the notebooks. No manual download is required. If you wish to download manually, see: https://huggingface.co/datasets/adsabs/SciX_UAT_keywords

## Setup and Usage

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/SciX-MultiLabel-Classification.git
cd SciX-MultiLabel-Classification

# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit the configuration files in `configs/` as needed. Example for loading config in Python:
```python
from src.utils import load_config
config = load_config('configs/base_config.yaml')
```

### Running Notebooks

To launch the Jupyter notebooks:
```bash
jupyter notebook
```
Open the desired notebook (e.g., `notebooks/03_model_training.ipynb`) in your browser.

### Training Models
1. **Data Preprocessing**:
```python
from src.data_preprocessing import ScientificTextPreprocessor
preprocessor = ScientificTextPreprocessor()
processed_data = preprocessor.preprocess_data(raw_data)
```

2. **Training Baseline**:
```python
from src.models import ImprovedMultiLabelModel
model = ImprovedMultiLabelModel(config)
model.train(X_train, y_train)
```

3. **Training Transformer**:
```python
from notebooks.transformer_model import train_transformer
model = train_transformer(train_dataset, val_dataset)
```

### Evaluation

After training, model artifacts are saved in the `models/` directory. For example:
- Baseline model: `models/baseline_model.pkl`
- Label binarizer: `models/label_binarizer.pkl`
- Processed validation data: `data/processed/val_processed.parquet`

Example evaluation code:
```python
from src.evaluate import evaluate_model
metrics = evaluate_model('models/baseline_model.pkl', 'models/label_binarizer.pkl', 'data/processed/val_processed.parquet')
```

## Results
- Baseline TF-IDF Model: 0.23 Micro-F1
- DistilBERT Model: 0.35 Micro-F1
- Best performance on well-represented classes
- Challenges with rare labels (< 10 samples)

## Future Improvements
- Implement label embeddings for better handling of rare classes
- Add data augmentation for underrepresented labels
- Experiment with larger transformer models
- Implement active learning for efficient labeling

## License
MIT License
