#!/bin/bash

# Function to display usage
usage()
{
    echo "Usage:"
    echo "   ./launch_container_custom.sh <commit> \n"
}

# Get timestamp
timestamp=$(date +"%Y-%m-%d_%H%M%S")

# Get commit
if [ -z "$1" ]; then
    commit=$(git rev-parse --verify HEAD)
    container_name="container.sif"
else
    commit=$(git rev-parse --verify "$1")
    container_name="container_${timestamp}_${commit}.sif"
fi

# Check if apptainer exists, otherwise use singularity
PLATFORM="Apptainer"
if ! command -v apptainer &> /dev/null; then
    PLATFORM="Singularity"
    alias apptainer=singularity
fi

# Create container if it does not exist
if [ ! -e ./apptainer/$container_name ]; then
    echo Build $PLATFORM image
    cmd="APPTAINERENV_GITLAB_TOKEN=$GITLAB_TOKEN APPTAINERENV_COMMIT=$commit apptainer build --fakeroot --force apptainer/$container_name apptainer/container.def"
    echo $cmd
    eval $cmd
fi

# Shell into container
if [ -z "$1" ]; then
    echo Run a shell within a container with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    cmd="APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES APPTAINERENV_WANDB_API_KEY=$WANDB_API_KEY apptainer shell --pwd /project/ --bind $(pwd):/project/ --cleanenv --containall --home /tmp/ --no-home --nv --workdir apptainer/ apptainer/$container_name"
    echo $cmd
    eval $cmd
fi
