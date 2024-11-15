#!/bin/bash

container_name='NeuralNetworkOnWeatherPrediction'

# Function to check if Docker container is running
container_running() {
    docker ps --filter "name=$container_name" --format '{{.Names}}' | grep -q "$container_name"
}

# Check if the container already exists
if docker ps -a --format '{{.Names}}' | grep -q $container_name; then
    echo "Container '$container_name' already exists."
    CONTAINER_ID=$(docker ps --format '{{.ID}}' --filter "name=$container_name")

    if [ "$(docker inspect --format '{{.State.Status}}' $CONTAINER_ID)" == "running" ]; then
        echo "Container is running, using the existing container..."
    else
        echo "Container is not running, starting it..."
        docker start $container_name
        # Introduce a delay until the container is healthy
        sleep 20
    fi
else
    echo "Container '$container_name' does not exist, creating it..."
    CONTAINER_ID=$(docker run --rm -d --name $container_name --privileged --entrypoint /bin/bash arm32v7/ubuntu:latest)
    echo "Container ID after creation: $CONTAINER_ID"

    sleep 20
fi

if [ -z "$CONTAINER_ID" ]; then
    echo "Failed to create or retrieve container ID. Exiting."
    exit 1
fi

# Print container ID
echo "** Container ID: $CONTAINER_ID"

# Check if the container exists before executing commands
if docker ps -a --format '{{.ID}}' | grep -q $CONTAINER_ID; then
    echo "Container exists, proceeding with package installation..."

    # Install necessary packages if they are not installed
    echo "Installing necessary packages..."
    docker exec $CONTAINER_ID bash -c 'which node npm python3 pip3 tsc curl wget' > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Installing necessary packages..."
        docker exec $CONTAINER_ID bash -c 'which python3' > /dev/null 2>&1 || docker exec $CONTAINER_ID apt-get install -y python3
        docker exec $CONTAINER_ID bash -c 'which pip3' > /dev/null 2>&1 || docker exec $CONTAINER_ID apt-get install -y python3-pip
        docker exec $CONTAINER_ID bash -c 'which curl' > /dev/null 2>&1 || docker exec $CONTAINER_ID apt-get install -y curl
        docker exec $CONTAINER_ID bash -c 'which wget' > /dev/null 2>&1 || docker exec $CONTAINER_ID apt-get install -y wget
        docker exec $CONTAINER_ID apt-get install -y --upgrade setuptools
        docker exec $CONTAINER_ID bash -c 'which numpy' > /dev/null 2>&1 || docker exec $CONTAINER_ID apt-get install -y RPI.GPIO
        docker exec $CONTAINER_ID bash -c 'which sklearn' > /dev/null 2>&1 || docker exec $CONTAINER_ID apt-get install -y adafruit-blinka
    else
        echo "Necessary packages are already installed."
    fi

    # Update and upgrade packages
    docker exec $CONTAINER_ID apt-get update -y
    docker exec $CONTAINER_ID apt-get upgrade -y

    # Clone Anemometer using find
    #if docker exec $CONTAINER_ID find /path/to/container/root -name 'anemometer' -type d | grep -q 'anemometer'; then
    #    echo "FIND - Anemometer is already cloned in the container."
    #else
    #    echo "FIND - Cloning Anemometer repository..."
    #    docker exec $CONTAINER_ID git clone https://github.com/jonathan-99/anemometer.git anemometer
    #fi


    # Remove Anemometer repository if already cloned
    docker exec $CONTAINER_ID ls anemometer &> /dev/null
    if [ $? -eq 0 ]; then
        echo "Anemometer repository already exists in the container. Removing..."
        docker exec $CONTAINER_ID rm -rf anemometer
    fi

    # Clone Anemometer repository
    echo "Cloning Anemometer repository..."
    docker exec $CONTAINER_ID git clone https://github.com/jonathan-99/anemometer.git anemometer


    # Print OS version
    echo "OS Version:"
    docker exec $CONTAINER_ID cat /etc/os-release

    # Print Python version
    echo "Python Version:"
    docker exec $CONTAINER_ID python3 --version

    # Print unittest version
    echo "unittest Version:"
    docker exec $CONTAINER_ID python3 -m unittest

    # Print coverage version
    echo "coverage Version:"
    docker exec $CONTAINER_ID coverage --version

    # Print Docker image ID
    echo "Docker Image ID:"
    docker exec $CONTAINER_ID cat /proc/self/cgroup | grep "docker" | sed 's/^.*\///' | head -n 1

else
    echo "Error: Container does not exist. Exiting."
    exit 1
fi