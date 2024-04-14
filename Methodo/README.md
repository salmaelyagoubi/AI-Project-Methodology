
# Project Methodo

### Installation

1. Click on the repository URL:
   - Navigate to the GitHub page of the project and locate the "Code" button.
   - Click on it and copy the repository URL.

2. Clone the repository:
   ```bash
   git clone [paste-the-copied-repository-url-here]
   cd Methodo

## Description
This project uses machine learning to predict churn. It is implemented using Python and integrates tools such as MLflow for managing experiments and Sphinx for generating project documentation.

## Setup
### Prerequisites
- Python 3.8 or higher
- Conda or Miniconda

### Installation
1. Clone the repository:
   ```
   git clone [repository-url]
   cd Methodo
   ```

2. Create a Conda environment and activate it:
   ```
   conda create -n methodo-env python=3.8
   conda activate methodo-env
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Initialize MLflow
MLflow is used to track experiments, including parameter tuning, metrics, and model logging.
- Start the MLflow tracking server:
  ```
  mlflow ui
  ```
- Access the MLflow UI by navigating to `http://localhost:5000` on your web browser.

### Data
Ensure that you have the necessary data in the `data` directory. Modify the data path in the scripts if necessary.

## Running the Project
Execute the main script to start the project:
```
python flow.py
```

## Documentation with Sphinx
To generate documentation:
1. Navigate to the `docs` directory:
   ```
   cd docs
   ```
2. Build the documentation using Sphinx:
   ```
   make html
   ```
3. The generated HTML files will be available in `docs/_build/html`. Open `index.html` in a web browser to view the documentation.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request.
