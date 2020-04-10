#!/bin/bash

export SINGULARITY_TMPDIR=/share/rcifdata/pmanek/singularity/tmpdir
export SINGULARITY_CACHEDIR=/share/rcifdata/pmanek/singularity/cache
export MAIN_DIR=/share/rcifdata/pmanek/fusion
export BATCH_TAG=$1
shift # TODO: make clear that this is a parameter
export RUN_DIR=${MAIN_DIR}/hyperopt/${BATCH_TAG}
export IMAGE_NAME=tbr_reg26.sif

export RUN_COMMAND=$(cat "${RUN_DIR}/search_command")

set -e

cp ${MAIN_DIR}/images/${IMAGE_NAME} /tmp/${IMAGE_NAME}

cd /tmp
echo "Running command: ${RUN_COMMAND}"
singularity exec -B /share:/share /tmp/${IMAGE_NAME} ${RUN_COMMAND}
echo "Command terminated"

rm -f /tmp/${IMAGE_NAME}
echo "Done"
