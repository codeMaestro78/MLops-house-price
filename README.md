# MLOps Starter — House Prices

This repository is a beginner-friendly end-to-end ML project for a house price regression problem.

Features:
- Data loading & preprocessing (scikit-learn)
- Training script with MLflow tracking & model registry
- FastAPI service to serve predictions
- Docker + docker-compose to run MLflow server and API locally
- GitHub Actions workflow for CI
- Minimal pytest unit tests

## Quickstart (local)

1. Create a Python venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
