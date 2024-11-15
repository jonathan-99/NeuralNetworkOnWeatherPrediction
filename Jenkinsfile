#! groovy

// This file is designed to provide a jenkins instance with a pipeline to download, unittest, coverage test this

// note jenkins / pipeline doesn't like running python through groovy, so they're extracted to

pipeline {
    agent any
    stages {
        stage('checks') {
            steps {
                script {
                    // need to add a check on the image
                    sh """
                        echo "this is the npm version ${sh(script: 'git --version', returnStdout: true).trim()}"
                        echo "this is the python version ${sh(script: 'python --version', returnStdout: true).trim()}"
                        echo "this is the ufw version ${sh(script: 'sudo ufw status', returnStdout: true).trim()}"
                        echo "this is the docker version ${sh(script: 'docker -v', returnStdout: true).trim()}"
                    """
                }
            }
        }
        stage('setup checks') {
            steps {
                script {
                    sh """
                        file=jenkins_scripts/setup.sh
                        chmod +x jenkins_scripts/setup.sh
                        ls -l jenkins_scripts/setup.sh
                        script -q -c ./jenkins_scripts/setup.sh /dev/null
                    """
                }
            }
        }
        stage('download and install docker image with dependencies') {
            steps {
                script {
                    sh """
                        file1='jenkins_scripts/download_and_install.sh'
                        chmod +x \$file1
                        filePermissions=\$(ls -l \$file1)
                        echo "File permissions: \$filePermissions"
                        script -q -c "./\$file1" /dev/null
                    """
                }
            }
        }
        stage('unittest, PEP8, coverage report') {
            steps {
                script {
                    sh """
                        file2='jenkins_scripts/do_unittests.sh'
                        chmod +x \$file2
                        filePermissions=\$(ls -l \$file2)
                        echo "File permissions: \$filePermissions"
                        script -q -c "./\$file2" output_tests_readme.txt
                    """
                }
            }
        }
        stage('stop docker container'){
            steps {
                script {
                    sh """

                    """
                }
            }
        }
    }
}