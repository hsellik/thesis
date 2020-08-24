import os
from argparse import ArgumentParser

from config import Config
from model import Model
from token_processor import TokenProcessor

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-s", "--save", dest="save_path",
                        help="path to save model", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to load model from", metavar="FILE", required=False)
    args = parser.parse_args()

    config = Config.get_default_config(args)

    model = Model(config)
    token_processor = TokenProcessor(config)

    if config.DATA_PATH and config.SAVE_PATH:
        token_processor.remove_rare_tokens()
        token_sequences = token_processor.get_token_sequences()
        model.fit(token_sequences)
    if config.DATA_PATH and config.LOAD_PATH:
        if not os.path.isfile(config.TEMP_DATA_PATH):
            token_processor.remove_rare_tokens()
        token_sequences = token_processor.get_token_sequences_for_evaluation()
        results = model.evaluate(token_sequences)
        for line in results:
            print(line)
