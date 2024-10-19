
# Azure End-to-End MLOps Project

This project demonstrates an end-to-end MLOps workflow using Azure Machine Learning services. It covers data preprocessing, model training, deployment, and scoring pipelines, integrated with Azure components like Workspaces, Datasets, Compute, and Pipelines.

```
├── data_wrangling.py      # Data preprocessing and Azure dataset handling
├── preprocessing.py       # Detailed data preprocessing using QuantileTransformer
├── modeling.py            # Model training, evaluation, and versioning
├── score.py               # Scoring script for deployed models
├── pipeline.ipynb         # Orchestrating the pipeline with AzureML
└── README.md              # Project documentation
```

## Features

- **Data Wrangling**: Data ingestion and preprocessing using AzureML Datasets and Datastores. Handles missing data, transformations, and dataset splitting for training and testing.
- **Preprocessing**: Detailed preprocessing using QuantileTransformer for normalization, handling missing values, and exporting preprocessed data to Azure Blob storage.
- **Modeling**: Train, evaluate, and save models using scikit-learn. Leverages AzureML Workspace and cloud compute targets for scalable ML workflows.
- **MLOps Pipeline**: An automated pipeline using AzureML services, from data preprocessing to deployment. Uses Azure compute resources for training and registers models for reproducibility.
- **Model Scoring**: A scoring script that loads a trained model and returns predictions for incoming data.

## Prerequisites

Before running this project, ensure you have the following:

- Azure subscription with access to AzureML services.
- Python 3.x installed.
- Azure CLI and `azureml-sdk` package.
- Necessary Python packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

```bash
pip install azureml-core pandas numpy matplotlib scikit-learn
```

## Running the Pipeline

1. **Data Wrangling**:
   Execute the `data_wrangling.py` script to preprocess the data:

   ```bash
   python data_wrangling.py --input data.csv --output preprocessed_data.csv
   ```

2. **Data Preprocessing**:
   Further preprocessing of the data using `preprocessing.py`:

   ```bash
   python preprocessing.py --prep preprocessed_data.csv
   ```

   This script:
   - Handles missing values by replacing 0s with means or medians for skewed distributions.
   - Applies quantile transformation for normalization.
   - Uploads the preprocessed data back to the Azure Blob datastore.

3. **Model Training**:
   Train the model using `modeling.py`:

   ```bash
   python modeling.py --data preprocessed_data.csv --model_output model.pkl
   ```

4. **Pipeline Execution**:
   Use the `pipeline.ipynb` notebook to configure and execute the end-to-end AzureML pipeline, automating the entire process from data preparation to model deployment.

## Model Scoring

Once the model is deployed, use the `score.py` script for predictions:

```bash
python score.py --data input_data.json
```

- This script loads a pre-trained model (retrieved from AzureML model registry) and provides predictions for the input data.
- Input data should be in JSON format, containing the features necessary for the model.

## Model Deployment

Once trained, models can be deployed using AzureML endpoints. Details on deploying models can be found in the pipeline notebook, where deployment steps are automated.

# Screenshot


