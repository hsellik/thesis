###########################################################
# Change the following values to train a new model.
# type: the name of the new model, only affects the saved file name.
# dataset: the name of the dataset, as was preprocessed using preprocess.sh
# test_data: by default, points to the validation set, since this is the set that
#   will be evaluated after each training iteration. If you wish to test
#   on the final (held-out) test set, change 'val' to 'test'.
# imbalanced: indicate whether to use best hyper-parameters for imbalanced or
#   balanced dataset, defaults to --no-imbalanced (enable with --imbalanced)
type=${1:-"code2seq_model_large"}
dataset_name=${2:-"java_dataset_code2seq"}
data_dir=data/${dataset_name}
data=${data_dir}/${dataset_name}
test_data=${data_dir}/${dataset_name}.val.c2s
model_dir=models/${type}
imbalanced=${3:-"--no-imbalanced"}

mkdir -p ${model_dir}
set -e
python3 -u code2seq.py --data ${data} --test ${test_data} ${imbalanced} --save_prefix ${model_dir}/model
