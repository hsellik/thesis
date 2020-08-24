import os
import subprocess

import wandb

from common import GPUSelector
from config import Config
from evaluator import Evaluator
from interactive_predict import InteractivePredictor
from model_base import Code2VecModelBase
from vocabularies import VocabType


def load_model_dynamically(config: Config) -> Code2VecModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}
    if config.DL_FRAMEWORK == 'tensorflow':
        from tensorflow_model import Code2VecModel
    elif config.DL_FRAMEWORK == 'keras':
        from keras_model import Code2VecModel
    return Code2VecModel(config)


if __name__ == '__main__':
    config = Config(set_defaults=False, load_from_args=True, verify=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUSelector(config).pick_gpu_lowest_memory())
    wandb.init(config=config, project="msc_thesis_hendrig")
    # PROCESSNIG FOR WANDB SWEEPING
    # config.update_values(wandb.config)
    # argument order DATASET_NAME=$1, REALISTIC_RATIO=$2, MAX_CONTEXTS=$3,  WORD_VOCAB_SIZE=$4, PATH_VOCAB_SIZE=$5
    if config.PREPROCESS:
        subprocess.call(
            ['sh', 'preprocess.sh', str(config.TRAIN_DATA_PATH_PREFIX.split("/")[-1]), str(config.REALISTIC_RATIO),
             str(config.MAX_CONTEXTS), str(config.MAX_TOKEN_VOCAB_SIZE), str(config.MAX_PATH_VOCAB_SIZE)])
    # END PROCESSING WANDB SWEEPING

    model = load_model_dynamically(config)
    config.log('Done creating code2vec model')

    if config.is_training:
        model.train()
    if config.SAVE_W2V:
        model.save_word2vec_format(config.SAVE_W2V, VocabType.Token)
        config.log('Origin word vectors saved in word2vec text format in: %s' % config.SAVE_W2V)
    if config.SAVE_T2V:
        model.save_word2vec_format(config.SAVE_T2V, VocabType.Target)
        config.log('Target word vectors saved in word2vec text format in: %s' % config.SAVE_T2V)
    if (config.is_testing and not config.is_training) or config.RELEASE:
        evaluator = Evaluator(config, model)
        evaluator.evaluate()
    if config.PREDICT:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    model.close_session()
