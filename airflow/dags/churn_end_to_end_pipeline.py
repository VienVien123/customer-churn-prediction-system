from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="churn_end_to_end_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["churn", "mlops"],
) as dag:

    ingest = BashOperator(
        task_id="ingest_raw_data",
        bash_command="cd /opt/project && python3 data-pipeline/scripts/ingest.py"
    )

    validate = BashOperator(
        task_id="validate_data",
        bash_command="cd /opt/project && python3 data-pipeline/scripts/validate.py data-pipeline/data/raw/latest.csv artifacts/reports/validation_latest.json"
    )

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="cd /opt/project && python3 data-pipeline/scripts/preprocess.py data-pipeline/data/raw/latest.csv data-pipeline/data/processed/processed_latest.parquet"
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="cd /opt/project && python3 data-pipeline/scripts/build_features.py data-pipeline/data/processed/processed_latest.parquet data-pipeline/data/features/features_latest.parquet"
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/project && python3 model_pipeline/src/train.py --input_path data-pipeline/data/features/features_latest.parquet --output_dir artifacts/models --mlflow_uri http://host.docker.internal:5000"
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="cd /opt/project && python3 model_pipeline/src/evaluate.py --input_path data-pipeline/data/features/features_latest.parquet --model_path artifacts/models/model.joblib --output_path artifacts/metrics/eval_latest.json"
    )

    register = BashOperator(
        task_id="register_model",
        bash_command="cd /opt/project && python3 model_pipeline/src/register.py"
    )

    deploy = BashOperator(
        task_id="deploy_model",
        bash_command="cd /opt/project && python3 model_pipeline/src/deploy.py"
    )

    ingest >> validate >> preprocess >> build_features >> train >> evaluate >> register >> deploy