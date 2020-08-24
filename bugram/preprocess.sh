#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# PROJECT_DIR - path to the project to evaluate
# EXTRACTOR_JAR - path to the tokenizer JAR
# NUM_THREADS - the number of parallel threads to use. It is
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# DATASET_NAME - name of the dataset to save tokens
PROJECT_DIR=""
EXTRACTOR_JAR="Tokenizer/build/libs/Tokenizer-0.0.1-SNAPSHOT.jar"
NUM_THREADS=32
DATASET_NAME="tokens.txt"
###########################################################
data_dir="data"
data_file_path=${data_dir}/${DATASET_NAME}

mkdir -p ${data_dir}
echo "Extracting tokens from project: ${PROJECT_DIR}"
java -cp ${EXTRACTOR_JAR} Tokenizer.App --dir ${PROJECT_DIR} --num_threads ${NUM_THREADS} > ${data_file_path} 2>> error_log.txt
echo "Finished extracting tokens"
