#!/usr/bin/env bash
###########################################################
# Calls preprocess.sh to create a dataset with more realistic
# bug ratio than default
# Change the following values to preprocess a new dataset.
# DATASET_NAME is just a name for the currently extracted
#   dataset.
# REALISTIC_RATIO is for generating 1 mutation per every 10 correct
#   examples for training data. Check the JavaExtractor source to be
#   if you wish to be 100% sure about the ratio.
###########################################################
DATASET_NAME="java_dataset_imbalanced"
REALISTIC_RATIO='true'
sh preprocess.sh ${DATASET_NAME} ${REALISTIC_RATIO}