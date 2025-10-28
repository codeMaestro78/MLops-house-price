# MLOps House Price Prediction

An end-to-end MLOps project for house price regression using scikit-learn, MLflow, and FastAPI.

## Features

- Data loading and preprocessing with scikit-learn
- Model training with MLflow tracking and model registry
- FastAPI service for real-time predictions
- Docker and docker-compose for local deployment
- GitHub Actions CI/CD pipeline
- Unit tests with pytest

## Prerequisites

- Python 3.11+
- Docker (optional)

## Quick Start (Local)

1. **Setup environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python -m src.train --experiment-name house-prices-demo
   ```

3. **Start MLflow server:**
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns
   ```

4. **Run the API:**
   ```bash
   uvicorn src.serve:app --host 0.0.0.0 --port 8000
   ```

5. **Test prediction:**
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"data": [[8, 41.0, 6.6, 0.0, 3.4, 6.3, 2.0, 1.0]]}'
   ```

## Docker Deployment

```bash
docker-compose up --build
```

This starts MLflow server on port 5000 and API on port 8000.

## Testing

```bash
pytest tests/
```

## Project Structure

- `src/`: Source code (data, model, train, serve)
- `tests/`: Unit tests
- `mlruns/`: MLflow artifacts
- `Dockerfile`: API container
- `docker-compose.yml`: Multi-service setup
