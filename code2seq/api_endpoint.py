from common import Common
from config import Config
from extractor import Extractor
import re
from flask import Flask, request, jsonify

SHOW_TOP_CONTEXTS = 5
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'java-extractor/build/libs/JavaExtractor-0.0.1-SNAPSHOT.jar'
FILE_NAME_INDEX = 2
METHOD_NAME_INDEX = 3


class APIEndpoint:

    app = Flask(__name__)
    @app.route('/ast/', methods=['POST'])
    def getPrediction():
        # Get response
        # Save code to file
        # call JAR to get model input
        # get model results
        # call JAR to get JSON
        # add styling from lowest to highest priority
        # return JSON
        content = request.json
        print(content)
        #self.getPrediction(self)
        return jsonify({"uuid":"sdf"})

    def __init__(self, config: Config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config, JAR_PATH, max_path_length=MAX_PATH_LENGTH, max_path_width=MAX_PATH_WIDTH)
        self.find_data_path = config.FIND_BUGS_PATH
        self.path_extractor = Extractor(self.config, JAR_PATH, max_path_length=MAX_PATH_LENGTH, max_path_width=MAX_PATH_WIDTH)


    def run(self):
        self.app.run(port=8080, host='127.0.0.1', debug=True)



   # def getPrediction2(self):
   #     with open(self.find_data_path, 'r') as file:
   #         for line in file.readlines():
   #             predict_line, pc_info_dict = self.path_extractor.process_line(line)
   #             model_results = self.model.predict(predict_line)
   #             prediction_results = Common.parse_results(model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS)
#
   #             for index, prob in enumerate(prediction_results[0].probs[0][0]):
   #                 print("Result")
   #                 #print(line.split(" ")[0])
   #                 java_file = line.split(' ')[0].split('#')[FILE_NAME_INDEX]
   #                 method_name = line.split(' ')[0].split('#')[METHOD_NAME_INDEX]
   #                 method_name = re.sub(r"(\|)([a-zA-z])", lambda match: match.group(2).upper(), method_name)
   #                 print('%.3f : %s' % (prob, self.model.index_to_target[index]))
   #                 print(f"{method_name}")
   #                 print(f"open -a XCode {java_file}")
#