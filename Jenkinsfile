pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI_PROD}"
        PYTHON_ENV_PATH = "venv/bin/python3"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/your-org/your-repo.git'
            }
        }
        stage('Setup Environment') {
            steps {
                sh "${PYTHON_ENV_PATH} -m venv venv"
                sh "source venv/bin/activate && pip install -r requirements.txt"
                sh "source venv/bin/activate && pip install -r requirements_cli.txt"
            }
        }
        stage('Security Scan') {
            steps {
                sh "source venv/bin/activate && pip install pip-audit"
                sh "source venv/bin/activate && pip-audit -r requirements.txt -r requirements_api.txt -r requirements_cli.txt"
            }
        }
        stage('Functional Tests') {
            steps {
                sh "source venv/bin/activate && PYTHONPATH=. pytest tests/functional/ -v -s --junitxml=jenkins-report.xml"
            }
        }
        stage('Run MLflow Pipeline') {
            steps {
                script {
                    def run_command = "${PYTHON_ENV_PATH} mlflow_enterprise_orchestrator.py"
                    try {
                        sh "source venv/bin/activate && ${run_command}"
                        // Optionally, add a step to check pipeline_summary.json for status
                        sh "source venv/bin/activate && cat results/mlflow_enterprise/enterprise_run_report.json"
                    } catch (e) {
                        currentBuild.result = 'FAILURE'
                        error "MLflow Pipeline failed: ${e}"
                    }
                }
            }
        }
        stage('Verify Model Performance') {
            steps {
                script {
                    // This stage would typically fetch metrics from MLflow and compare against a threshold
                    // Example: check a specific metric from the last run
                    // For this example, we assume the orchestrator writes a success/failure status
                    def report_content = sh(script: "cat results/mlflow_enterprise/enterprise_run_report.json", returnStdout: true).trim()
                    def report_json = readJSON text: report_content
                    
                    if (report_json.status == "FAILURE") {
                        currentBuild.result = 'FAILURE'
                        error "Model performance did not meet CI/CD thresholds."
                    }
                }
            }
        }
        stage('Model Serving Deployment (Optional)') {
            steps {
                // This would trigger a deployment to a model serving platform
                // For demonstration, just log a message
                echo "Model serving deployment triggered for latest successful model."
                // Example: sh "mlflow models deploy -m runs:/${LAST_MLFLOW_RUN_ID}/model --target sagemaker"
            }
        }
    }
    post {
        always {
            cleanWs()
        }
    }
}
