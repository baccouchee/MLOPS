# mlops-project/mlops-project/README.md

# MLOps Project

This project implements a machine learning pipeline using Docker and Jenkins for continuous integration and deployment (CI/CD). The pipeline includes data downloading, cleaning, and model training.

## Project Structure

```
mlops-project
├── data
│   ├── train.csv          # Raw training data
│   ├── train_clean.csv    # Cleaned training data
│   ├── test.csv           # Raw test data
│   └── test_clean.csv     # Cleaned test data
├── scripts
│   ├── download_data.sh    # Script to download datasets
│   ├── clean_data.sh       # Script to clean data using Docker
│   └── train_model.sh      # Script to train the model
├── Dockerfile               # Dockerfile for building the application image
├── docker-compose.yml       # Docker Compose file for managing services
├── Jenkinsfile              # Jenkins pipeline configuration
└── requirements.txt         # Python dependencies
```

## Setup Instructions

1. **Clone the repository:**

   ```
   git clone <repository-url>
   cd mlops-project
   ```

2. **Build the Docker image:**

   ```
   docker-compose build
   ```

3. **Run the application:**
   ```
   docker-compose up
   ```

## Usage

- To download the data, run:

  ```
  ./scripts/download_data.sh
  ```

- To clean the data, run:

  ```
  ./scripts/clean_data.sh
  ```

- To train the model, run:
  ```
  ./scripts/train_model.sh
  ```

## Jenkins Integration

This project includes a Jenkins pipeline defined in the `Jenkinsfile`. The pipeline automates the following steps:

1. Downloading the datasets.
2. Cleaning the data using the Dockerized environment.
3. Training the model and storing the generated artifacts.

## Requirements

Ensure you have the following installed:

- Docker
- Docker Compose
- Jenkins (for CI/CD)

## License

This project is licensed under the MIT License.
