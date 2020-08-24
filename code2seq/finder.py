from common import Common
from config import Config
from extractor import Extractor
import re

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'java-extractor/build/libs/JavaExtractor-0.0.1-SNAPSHOT.jar'
PROBABILITY_THRESHOLD = 0.8
FILE_NAME_INDEX = 2
METHOD_NAME_INDEX = 3


class Finder:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config: Config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config, JAR_PATH, max_path_length=MAX_PATH_LENGTH, max_path_width=MAX_PATH_WIDTH)
        self.find_data_path = config.FIND_BUGS_PATH

    def find(self):
        path_extractor = Extractor(self.config, JAR_PATH, max_path_length=MAX_PATH_LENGTH, max_path_width=MAX_PATH_WIDTH)
        total = 0
        potential_bug = 0
        with open(self.find_data_path, 'r') as file:
            for line in file.readlines():
                total += 1
                predict_line, pc_info_dict = path_extractor.process_line(line)
                model_results = self.model.predict(predict_line)

                prediction_results = Common.parse_results(model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS)

                for index, prob in enumerate(prediction_results[0].probs[0][0]):
                    if prob > PROBABILITY_THRESHOLD and self.model.index_to_target[index] == "bug" and "test" not in line.split(" ")[0]:
                        potential_bug += 1
                        print("BUG?!")
                        #print(line.split(" ")[0])
                        java_file = line.split(' ')[0].split('#')[FILE_NAME_INDEX]
                        method_name = line.split(' ')[0].split('#')[METHOD_NAME_INDEX]
                        method_name = re.sub(r"(\|)([a-zA-z])", lambda match: match.group(2).upper(), method_name)
                        print('%.3f : %s' % (prob, self.model.index_to_target[index]))
                        print(f"{method_name}")
                        print(f"open -a XCode {java_file}")
                if total % 100 == 0:
                    print(f"Total: {total}, potential bug: {potential_bug}")

        print(f"Total: {total}, potential bug: {potential_bug}")