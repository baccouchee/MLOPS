pipeline {
    agent any

    stages {
        stage('Clean Data') {
            steps {
                script {
                    echo "Starting data cleaning..."
                    // Build and run the Docker container to clean data
                    try {
                        bat 'docker run data'
                        echo "Data cleaning completed."
                    } catch (Exception e) {
                        echo "Data cleaning failed: ${e}"
                        currentBuild.result = 'FAILURE'
                        error "Stopping pipeline due to failure in data cleaning."
                    }
                }
            }
        }
        stage('Train Model') {
            steps {
                script {
                    echo "Starting model training..."
                    // Build and run the Docker container to train the model
                    try {
                        bat 'docker-compose run app python scripts/modeling_&_evaluation.py'
                        echo "Model training completed."
                    } catch (Exception e) {
                        echo "Model training failed: ${e}"
                        currentBuild.result = 'FAILURE'
                        error "Stopping pipeline due to failure in model training."
                    }
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'model/*.pkl', allowEmptyArchive: true
            archiveArtifacts artifacts: 'logs/*.log', allowEmptyArchive: true
            archiveArtifacts artifacts: 'metrics/*.json', allowEmptyArchive: true
        }
        failure {
            script {
                echo "Pipeline failed. Check the logs for details."
            }
        }
    }
}