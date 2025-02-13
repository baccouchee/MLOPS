pipeline {
    agent any

    stages {

        stage('Clean Data') {
            steps {
                script {
                    // Build and run the Docker container to clean data
                    sh 'docker-compose run app python scripts/cleaning_data.py'
                }
            }
        }
        stage('Train Model') {
            steps {
                script {
                    // Build and run the Docker container to train the model
                    sh 'docker-compose run app python scripts/modeling_&_evaluation.py'
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