import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf

from api_endpoint import APIEndpoint
from common import GPUSelector
from config import Config
from evaluator import Evaluator
from finder import Finder
from interactive_predict import InteractivePredictor
from model import Model
import wandb


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)
    parser.add_argument('--imbalanced', dest='imbalanced', action='store_true',
                        help="Use hyper-parameters for imbalanced dataset")
    parser.add_argument('--no-imbalanced', dest='imbalanced', action='store_false',
                        help="Use hyper-parameters for balanced dataset")
    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a smaller model '
                             'size.')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--find_bugs', action='store_true')
    parser.add_argument("--find_data_path", dest="find_data_path",
                        help="path to find bugs from", required=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--api_endpoint', dest="api_endpoint", action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    args, unknown = parser.parse_known_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    print("starting to set arguments from javaparser")
    if args.api_endpoint:
        print("Setting default config")
        config = Config.get_default_config(args)
    elif args.imbalanced:
        config = Config.get_imbalanced_off_by_one_config(args)
    else:
        config = Config.get_balanced_off_by_one_config(args)

    wandb.init(config=config, project="msc_thesis_hendrig")
    #config.update_values(wandb.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUSelector(config).pick_gpu_lowest_memory())

    model = Model(config)
    print('Created model')
    if config.TRAIN_PATH:
        model.train()
    if config.TEST_PATH and not args.data_path:
        evaluator = Evaluator(config, model)
        evaluator.evaluate()
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if args.find_bugs and args.find_data_path:
        finder = Finder(config, model)
        finder.find()
    if args.release and args.load_path:
        model.evaluate(release=True)
    if args.api_endpoint:
        api_endpoint = APIEndpoint(config, model)
        api_endpoint.run()
    model.close_session()
