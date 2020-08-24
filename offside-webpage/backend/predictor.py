from extractor import Extractor
from common import Common

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'libs/JavaExtractor-0.0.1-SNAPSHOT.jar'


class Predictor:

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config, JAR_PATH, max_path_length=MAX_PATH_LENGTH,
                                        max_path_width=MAX_PATH_WIDTH)

    def predict(self, input_filename):
        print(f"Predicting for file: {input_filename}")
        predict_lines = ""
        try:
            predict_lines, pc_info_dict, original_json = self.path_extractor.extract_paths(input_filename)
        except ValueError:
            print("Error making prediction")
            return {"error": "Encountered error while extracing paths. Method not compiling or not containing any of the following binary operators (<;<=;>;>=)."}
        model_results = self.model.predict(predict_lines)

        response = {}
        prediction_results = Common.parse_results(model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS)
        for index, method_prediction in prediction_results.items():
            print('Original name:\t' + method_prediction.original_name)
            if self.config.BEAM_WIDTH == 0:
                print('Predicted:\t%s' % [step.prediction for step in method_prediction.predictions])
                print('Probabilities:')
                for index, prob in enumerate(method_prediction.probs[0][0]):
                    if prob > 0.001:
                        response[self.model.index_to_target[index]] = str(prob)
                        print(f"{prob} : {self.model.index_to_target[index]}")
                for timestep, single_timestep_prediction in enumerate(method_prediction.predictions):
                    print('Attention:')
                    print('TIMESTEP: %d\t: %s' % (timestep, single_timestep_prediction.prediction))
                    for inner_index, attention_obj in enumerate(single_timestep_prediction.attention_paths):
                        response[f"context{inner_index}"] = [str(attention_obj['score']), attention_obj['token1'],
                                                             attention_obj['path'],
                                                             attention_obj['token2']]
                        print('%f\tcontext: %s,%s,%s' % (
                            attention_obj['score'], attention_obj['token1'], attention_obj['path'],
                            attention_obj['token2']))
        merged_response = {**response, **original_json}
        del merged_response["modelInput"]
        return merged_response
