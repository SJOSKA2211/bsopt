pipeline {
    agent any

    environment {
        DOCKER_BUILDKIT = '1'
        COMPOSE_INTERACTIVE_NO_CLI = '1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Core Images') {
            steps {
                sh 'make build'
            }
        }

        stage('Lint & Security Scan') {
            parallel {
                stage('Linting') {
                    steps {
                        sh 'docker compose run --rm api ruff check .'
                        sh 'docker compose run --rm api black --check .'
                    }
                }
                stage('Security Audit') {
                    steps {
                        sh 'docker compose run --rm api pip-audit -r requirements.txt || true'
                        sh 'docker compose run --rm api bandit -r src/'
                    }
                }
            }
        }

        stage('Integration Tests') {
            steps {
                sh 'make test-all'
            }
        }

        stage('Model Verification (ML Profile)') {
            steps {
                sh 'docker compose --profile ml run --rm ray-head python src/ml/distributed_training.py --verify-only'
            }
        }

        stage('Deploy (Unified)') {
            when {
                branch 'main'
            }
            steps {
                # Deploy the full stack including proxy and observability
                sh './deploy.sh deploy --full'
            }
        }
    }

    post {
        always {
            sh 'make down || true'
            cleanWs()
        }
    }
}
