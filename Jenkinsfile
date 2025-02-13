pipeline {
    agent any

    stages {
        stage('Clean Data') {
            steps {
                script {
                    echo "Starting data cleaning..."
                    // Build and run the Docker container to clean data
                    sh 'docker-compose run app python scripts/cleaning_data.py'
                    echo "Data cleaning completed."
                }
            }
        }
        stage('Train Model') {
            steps {
                script {
                    echo "Starting model training..."
                    // Build and run the Docker container to train the model
                    sh 'docker-compose run app python scripts/modeling_&_evaluation.py'
                    echo "Model training completed."
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