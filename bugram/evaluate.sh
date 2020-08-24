###########################################################
# Change the following values to fit a new model.
# MODEL_NAME: the name of the new model, only affects the saved file name.
# DATASET_NAME: the name of the dataset, as was preprocessed using preprocess.sh
DATASET_NAME="tokens.txt"
MODEL_NAME="bugram.pkl"
###########################################################
data_dir="data"
data_file_path=${data_dir}/${DATASET_NAME}

model_dir="models"
model_file_path=${model_dir}/${MODEL_NAME}

set -e
echo "Finding bugs with model ${model_file_path} from file ${data_file_path}"
python3 -u bugram.py --data ${data_file_path} --load ${model_file_path}
echo "Finished"