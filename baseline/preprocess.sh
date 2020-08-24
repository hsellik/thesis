#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# INPUT_DATASET_PREFIX - path prefix where training and testing sets are
# TRAIN_DIR - training set dir
# TEST_DIR - testing set dir
# EXTRACTOR_JAR - path to the tokenizer JAR
# NUM_THREADS - the number of parallel threads to use. It is
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# REALISTIC_RATIO is for generating 1 mutation per every 10 correct
#   examples for training data.
# DATA_DIR - directory where token files will be saved
# TRAIN_DATA_FILE - name of the file where training data tokens
# will be saved
# TEST_DATA_FILE - name of the file where testing data tokens
# will be saved
INPUT_DATASET_PREFIX="./data/java-large/"
TRAIN_DIR="${INPUT_DATASET_PREFIX}/training"
TEST_DIR="${INPUT_DATASET_PREFIX}/testing"
EXTRACTOR_JAR="Tokenizer/build/libs/Tokenizer-0.0.1-SNAPSHOT.jar"
NUM_THREADS=32
REALISTIC_RATIO='false'
###########################################################
DATA_DIR="./data"
TRAIN_DATA_FILE="${DATA_DIR}/tokens_balanced_train.txt"
TEST_DATA_FILE="${DATA_DIR}/tokens_balanced_test.txt"

mkdir -p ${DATA_DIR}
echo "Extracting tokens for training set"
java -cp ${EXTRACTOR_JAR} Tokenizer.App --dir ${TRAIN_DIR} --realistic_bug_ratio ${REALISTIC_RATIO} --print_paths false --num_threads ${NUM_THREADS} > ${TRAIN_DATA_FILE} 2>> error_log.txt
echo "Extracting tokens for testing set"
java -cp ${EXTRACTOR_JAR} Tokenizer.App --dir ${TEST_DIR} --realistic_bug_ratio ${REALISTIC_RATIO} --print_paths false --num_threads ${NUM_THREADS} > ${TEST_DATA_FILE} 2>> error_log.txt
echo "Finished extracting tokens"
