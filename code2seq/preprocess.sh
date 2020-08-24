#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# TRAIN_DIR, VAL_DIR and TEST_DIR should be paths to
#   directories containing sub-directories with .java files
# INPUT_DATASET_PREFIX is a prefix to easily switch between different datasets
# OUTPUT_DATASET_NAME is just a name for the currently extracted
#   dataset.
# OFF_BY_ONE is for generating of-by-one mutations to training data
# NULLPOINTER is for generating nullpointer mutations to training data
# ONLY_OFF_BY_ONE_FEATURES is for generating training data with AST paths
#   which only pass through nodes containing less, lessEquals, greater or
#   greaterEquals comparison.
# REALISTIC_RATIO is for generating 1 mutation per every 10 correct
#   examples for training data. Check the JavaExtractor source to be
#   if you wish to be 100% sure about the ratio.
# MAX_DATA_CONTEXTS is the number of contexts to keep in the dataset for each
#   method (by default 1000). At training time, these contexts
#   will be downsampled dynamically to MAX_CONTEXTS.
# MAX_CONTEXTS - the number of actual contexts (by default 200) 
# that are taken into consideration (out of MAX_DATA_CONTEXTS)
# every training iteration. To avoid randomness at test time, 
# for the test and validation sets only MAX_CONTEXTS contexts are kept 
# (while for training, MAX_DATA_CONTEXTS are kept and MAX_CONTEXTS are
# selected dynamically during training).
# SUBTOKEN_VOCAB_SIZE, TARGET_VOCAB_SIZE -   
#   - the number of subtokens and target words to keep 
#   in the vocabulary (the top occurring words and paths will be kept). 
# NUM_THREADS - the number of parallel threads to use. It is 
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# PYTHON - python3 interpreter alias.
INPUT_DATASET_PREFIX=${1:-"./data/java-large/"}
TRAIN_DIR="${INPUT_DATASET_PREFIX}training"
VAL_DIR="${INPUT_DATASET_PREFIX}validation"
TEST_DIR="${INPUT_DATASET_PREFIX}testing"
#${nr:-value} used to parse arguments in order of entry or fall back to "value"
OUTPUT_DATASET_NAME=${2:-"java_dataset_code2seq"}
# Currently, OFF_BY_ONE and NULLPOINTER can not be true at the same time
OFF_BY_ONE='true'
NULLPOINTER='false'
ONLY_OFF_BY_ONE_FEATURES='false'
REALISTIC_RATIO=${3:-'false'}
MAX_DATA_CONTEXTS=1000
# taking max_contexts in excess for parameter tuning
# using less contexts while training
MAX_CONTEXTS=300
# same for vocab size
SUBTOKEN_VOCAB_SIZE=260000
TARGET_VOCAB_SIZE=2 #binary
NUM_THREADS=64
JAVA=java
PYTHON=python3
###########################################################

TRAIN_DATA_FILE=${OUTPUT_DATASET_NAME}.train.raw.txt
VAL_DATA_FILE=${OUTPUT_DATASET_NAME}.val.raw.txt
TEST_DATA_FILE=${OUTPUT_DATASET_NAME}.test.raw.txt
EXTRACTOR_JAR=java-extractor/build/libs/JavaExtractor-0.0.1-SNAPSHOT.jar

mkdir -p data
mkdir -p data/${OUTPUT_DATASET_NAME}

echo "Extracting paths from validation set..."
${JAVA} -cp ${EXTRACTOR_JAR} JavaExtractor.App --code2seq true --num_threads ${NUM_THREADS} --max_path_length 8 --max_path_width 2 --dir ${VAL_DIR} --off_by_one ${OFF_BY_ONE} --nullpointer ${NULLPOINTER} --realistic_bug_ratio ${REALISTIC_RATIO} --only_off_by_one_features ${ONLY_OFF_BY_ONE_FEATURES} > ${VAL_DATA_FILE} 2>> error_log.txt
#${PYTHON} JavaExtractor/extract.py --dir ${VAL_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} --off_by_one ${OFF_BY_ONE} --nullpointer ${NULLPOINTER} --realistic_bug_ratio true > ${VAL_DATA_FILE} 2>> error_log.txt
echo "Finished extracting paths from validation set"
echo "Extracting paths from test set..."
${JAVA} -cp ${EXTRACTOR_JAR} JavaExtractor.App --code2seq true --num_threads ${NUM_THREADS} --max_path_length 8 --max_path_width 2 --dir ${TEST_DIR} --off_by_one ${OFF_BY_ONE} --nullpointer ${NULLPOINTER} --realistic_bug_ratio ${REALISTIC_RATIO} --only_off_by_one_features ${ONLY_OFF_BY_ONE_FEATURES} > ${TEST_DATA_FILE} 2>> error_log.txt
#${PYTHON} JavaExtractor/extract.py --dir ${TEST_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} --off_by_one ${OFF_BY_ONE} --nullpointer ${NULLPOINTER} --realistic_bug_ratio true > ${TEST_DATA_FILE} 2>> error_log.txt
echo "Finished extracting paths from test set"
echo "Extracting paths from training set..."
${JAVA} -cp ${EXTRACTOR_JAR} JavaExtractor.App --code2seq true --num_threads ${NUM_THREADS} --max_path_length 8 --max_path_width 2 --dir ${TRAIN_DIR} --off_by_one ${OFF_BY_ONE} --nullpointer ${NULLPOINTER} --realistic_bug_ratio ${REALISTIC_RATIO} --only_off_by_one_features ${ONLY_OFF_BY_ONE_FEATURES} > ${TRAIN_DATA_FILE} 2>> error_log.txt
#${PYTHON} JavaExtractor/extract.py --dir ${TRAIN_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} --off_by_one ${OFF_BY_ONE} --nullpointer ${NULLPOINTER} --realistic_bug_ratio true  > ${TRAIN_DATA_FILE} 2>> error_log.txt
echo "Finished extracting paths from training set"
echo "Shuffling paths from training set"
${PYTHON} -c "import random; lines = open('${TRAIN_DATA_FILE}').readlines(); random.shuffle(lines); open('${TRAIN_DATA_FILE}', 'w').writelines(lines)"
echo "Finished shuffling paths from training set"

TARGET_HISTOGRAM_FILE=data/${OUTPUT_DATASET_NAME}/${OUTPUT_DATASET_NAME}.histo.tgt.c2s
SOURCE_SUBTOKEN_HISTOGRAM=data/${OUTPUT_DATASET_NAME}/${OUTPUT_DATASET_NAME}.histo.ori.c2s
NODE_HISTOGRAM_FILE=data/${OUTPUT_DATASET_NAME}/${OUTPUT_DATASET_NAME}.histo.node.c2s

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${SOURCE_SUBTOKEN_HISTOGRAM}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${NODE_HISTOGRAM_FILE}

${PYTHON} preprocess.py --train_data ${TRAIN_DATA_FILE} --test_data ${TEST_DATA_FILE} --val_data ${VAL_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --max_data_contexts ${MAX_DATA_CONTEXTS} --subtoken_vocab_size ${SUBTOKEN_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --subtoken_histogram ${SOURCE_SUBTOKEN_HISTOGRAM} \
  --node_histogram ${NODE_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} --output_name data/${OUTPUT_DATASET_NAME}/${OUTPUT_DATASET_NAME}

# If all went well, the raw data files can be deleted, because preprocess.py creates new files
# with truncated and padded number of paths for each example.
rm ${TRAIN_DATA_FILE} ${VAL_DATA_FILE} ${TEST_DATA_FILE} ${TARGET_HISTOGRAM_FILE} ${SOURCE_SUBTOKEN_HISTOGRAM} \
  ${NODE_HISTOGRAM_FILE}

