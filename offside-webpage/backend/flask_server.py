import os
import pathlib
import uuid
from argparse import ArgumentParser

from flask import Flask, request, Response, jsonify

from config import Config
from model import Model
from predictor import Predictor

SHOW_TOP_CONTEXTS = 5
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'java-extractor/libs/JavaExtractor-0.0.1-SNAPSHOT.jar'
FILE_NAME_INDEX = 2
METHOD_NAME_INDEX = 3

app = Flask(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-l", "--load", dest="load_path", help="path to saved file", metavar="FILE", required=False)
    args, unknown = parser.parse_known_args()
    print("Setting default config")
    config = Config.get_default_config(args)
    model = Model(config)
    predictor = Predictor(config, model)


    def save_to_file(code: str):
        file_name = str(uuid.uuid4()) + '.java'
        print(f"saving to file: {file_name}")
        file_path = os.path.join(pathlib.Path().absolute(), "temp", file_name)
        with open(file_path, 'w') as file:
            file.write(code)
        return file_path


    def get_model_results(file_name: str):
        print("Calling model")
        return predictor.predict(file_name)

    def color_node(node, key, value, color, stroke_width):
        if isinstance(node, list):
            for i in node:
                color_node(i, key, value, color, stroke_width)
        elif isinstance(node, dict):
            if key in node and str(node[key]) == value:
                node["color"] = "blue"
                node["nodeSvgShape"] = {
                    "shape": 'rect',
                    "shapeProps": {
                        "width": 120,
                        "height": 60,
                        "x": -60,
                        "y": 0,
                        "stroke": color,
                        "strokeWidth": stroke_width
                    }
                }
                return
            if "children" in node:
                for child in node["children"]:
                    color_node(child, key, value, color, stroke_width)

    colors = ["crimson", "coral", "salmon", "gold", "lightgoldenrodyellow"]

    def color_nodes(json_result: dict):
        all_paths = json_result['paths']
        for index in reversed(range(4)):
            short_path = json_result[f'context{index}'][2]
            for path in all_paths:
                if path['mainPath'] == short_path:
                    print(path['idPath'])
                    for id in path['idPath'].strip('][').split(', '):
                        color = f"{colors[index]}"
                        stroke_width = 7 - index
                        color_node(json_result['ast'], 'id', id, color, stroke_width)

    @app.route('/predict/', methods=['POST'])
    def getPrediction():
        # accept request contet
        content = request.json
        print("Predicting for code: ")
        print(content['code'])
        # save to temp file
        file_path = save_to_file(content['code'])
        # get model results
        print("Getting model results")
        json_result = get_model_results(file_path)
        # color nodes
        print("Got model results:")
        print(json_result)
        color_nodes(json_result)
        # delete temp file
        os.remove(file_path)
        # return JSON
        if "error" in json_result:
            return jsonify(json_result), 400
        else:
            return jsonify(json_result), 200


    app.run(port=8080, host='127.0.0.1', debug=True)
