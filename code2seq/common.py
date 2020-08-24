import re
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from config import Config


class Common:
    internal_delimiter = '|'
    SOS = '<S>'
    EOS = '</S>'
    PAD = '<PAD>'
    UNK = '<UNK>'

    @staticmethod
    def normalize_word(word):
        stripped = re.sub(r'[^a-zA-Z]', '', word)
        if len(stripped) == 0:
            return word.lower()
        else:
            return stripped.lower()

    @staticmethod
    def load_histogram(path, max_size=None):
        histogram = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                parts = line.split(' ')
                if not len(parts) == 2:
                    continue
                histogram[parts[0]] = int(parts[1])
        sorted_histogram = [(k, histogram[k]) for k in sorted(histogram, key=histogram.get, reverse=True)]
        return dict(sorted_histogram[:max_size])

    @staticmethod
    def load_vocab_from_dict(word_to_count, add_values=[], max_size=None):
        word_to_index, index_to_word = {}, {}
        current_index = 0
        for value in add_values:
            word_to_index[value] = current_index
            index_to_word[current_index] = value
            current_index += 1
        sorted_counts = [(k, word_to_count[k]) for k in sorted(word_to_count, key=word_to_count.get, reverse=True)]
        limited_sorted = dict(sorted_counts[:max_size])
        for word, count in limited_sorted.items():
            word_to_index[word] = current_index
            index_to_word[current_index] = word
            current_index += 1
        return word_to_index, index_to_word, current_index

    @staticmethod
    def binary_to_string(binary_string):
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list):
        return [Common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(binary_string_matrix):
        return [Common.binary_to_string_list(l) for l in binary_string_matrix]

    @staticmethod
    def binary_to_string_3d(binary_string_tensor):
        return [Common.binary_to_string_matrix(l) for l in binary_string_tensor]

    @staticmethod
    def legal_method_names_checker(name):
        return not name in [Common.UNK, Common.SOS, Common.PAD, Common.EOS]

    @staticmethod
    def filter_impossible_names(top_words):
        result = list(filter(Common.legal_method_names_checker, top_words))
        return result

    @staticmethod
    def unique(sequence):
        unique = []
        [unique.append(item) for item in sequence if item not in unique]
        return unique

    @staticmethod
    def parse_results(result, pc_info_dict, topk=5):
        prediction_results = {}
        results_counter = 0
        for single_method in result:
            original_name, top_suggestions, top_scores, attention_per_context, probs = list(single_method)
            current_method_prediction_results = PredictionResults(original_name, probs)
            if attention_per_context is not None:
                word_attention_pairs = [(word, attention) for word, attention in
                                        zip(top_suggestions, attention_per_context) if
                                        Common.legal_method_names_checker(word)]
                for predicted_word, attention_timestep in word_attention_pairs:
                    current_timestep_paths = []
                    for context, attention in [(key, attention_timestep[key]) for key in
                                               sorted(attention_timestep, key=attention_timestep.get, reverse=True)][
                                              :topk]:
                        if context in pc_info_dict:
                            pc_info = pc_info_dict[context]
                            current_timestep_paths.append((attention.item(), pc_info))

                    current_method_prediction_results.append_prediction(predicted_word, current_timestep_paths)
            else:
                for predicted_seq in top_suggestions:
                    filtered_seq = [word for word in predicted_seq if Common.legal_method_names_checker(word)]
                    current_method_prediction_results.append_prediction(filtered_seq, None)

            prediction_results[results_counter] = current_method_prediction_results
            results_counter += 1
        return prediction_results

    @staticmethod
    def compute_bleu(ref_file_name, predicted_file_name):
        with open(predicted_file_name) as predicted_file:
            pipe = subprocess.Popen(["perl", "scripts/multi-bleu.perl", ref_file_name], stdin=predicted_file,
                                    stdout=sys.stdout, stderr=sys.stderr)


class PredictionResults:
    def __init__(self, original_name, probs):
        self.original_name = original_name
        self.predictions = list()
        self.probs = probs

    def append_prediction(self, name, current_timestep_paths):
        self.predictions.append(SingleTimeStepPrediction(name, current_timestep_paths))

class SingleTimeStepPrediction:
    def __init__(self, prediction, attention_paths):
        self.prediction = prediction
        if attention_paths is not None:
            paths_with_scores = []
            for attention_score, pc_info in attention_paths:
                path_context_dict = {'score': attention_score,
                                     'path': pc_info.shortPath,
                                     'token1': pc_info.token1,
                                     'token2': pc_info.token2}
                paths_with_scores.append(path_context_dict)
            self.attention_paths = paths_with_scores


class PathContextInformation:
    def __init__(self, context):
        self.token1 = context['name1']
        self.shortPath = context['shortPath']
        self.token2 = context['name2']

    def __str__(self):
        return '%s,%s,%s' % (self.token1, self.shortPath, self.token2)


class Plots:
    def __init__(self, config: Config):
        self.config = config

    def plot_confusion_matrices(self, y_pred: list, y_true: list, axs: list = None):
        if axs is None:
            _, axs = plt.subplots(1, 2, figsize=(12, 4))
        plot_options = [("confusion_matrix", "Confusion Matrix", None, ".6g"),
                        ("confusion_matrix_normalized", "Normalized Confusion Matrix", 'all', ".2g")]
        index = 0
        for file_name, plot_name, normalize_option, values_format in plot_options:
            cm = confusion_matrix(y_true, y_pred, labels=[self.config.BUG_STRING, self.config.NO_BUG_STRING],
                                  normalize=normalize_option)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[self.config.BUG_STRING, self.config.NO_BUG_STRING])
            disp.plot(include_values=True, cmap='Blues', ax=axs[index], values_format=values_format,
                      xticks_rotation='horizontal')
            axs[index].title.set_text(plot_name)
            index += 1

    def plot_precision_recall_curve(self, y_prob: list, y_true: list, ax1: SubplotBase = None):
        # use sk-learn to get precision and recall fro different thresholds
        lr_precision, lr_recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=self.config.BUG_STRING)
        auc_score = auc(lr_recall, lr_precision)
        # set up plot
        if ax1 is None:
            _, ax1 = plt.subplots()
        ax1.title.set_text(f'Precision-Recall Curve \n with Area Under Curve: {auc_score:.2f}')
        ax2 = ax1.twinx()
        # draw lines and add legends
        ax1.plot(lr_recall, lr_precision, color='blue', marker=',', label='Precision-Recall')
        ax1.legend(loc=1)
        ax2.plot(lr_recall[:-1], thresholds, color='green', marker=',', label='Threshold-Recall')
        ax2.legend(loc=2)
        no_skill = y_true.count('bug') / len(y_true)
        ax1.plot([0, 1], [no_skill, no_skill], color='black', linestyle='--', label='No Skill')
        # set labels
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision', color='b')
        ax2.set_ylabel('Threshold', color='g')
        # match the values on y axes
        ylim = [0, 1.2]
        yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        ax1.set_ylim(ylim)
        ax1.set_yticks(yticks)
        ax2.set_ylim(ylim)
        ax2.set_yticks(yticks)

    def plot_roc_curve(self, y_prob: list, y_true: list, ax1: SubplotBase = None):
        # use sk-learn to get false positive rate and true positive rate for different thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=self.config.BUG_STRING)
        auc_score = auc(fpr, tpr)
        # set up plot
        if ax1 is None:
            _, ax1 = plt.subplots()
        ax1.title.set_text(f'Receiver Operating Characteristic Curve \n with Area Under Curve: {auc_score:.2f}')
        ax2 = ax1.twinx()
        # draw lines and legends
        ax1.plot(fpr, tpr, color='blue', marker=',', label='ROC')
        ax1.legend(loc=1)
        # last of fpr is 1 (ignored) and first of thresholds is not an actual threshold (ignored)
        ax2.plot(fpr[:-1], thresholds[1:], color='green', marker=',', label='Threshold-FP')
        ax2.legend(loc=2)
        plt.plot([0, 1], [0, 1], 'k--')
        # set labels
        ax1.set_xlabel('False positive rate')
        ax1.set_ylabel('True positive rate', color='b')
        ax2.set_ylabel('Threshold', color='g')
        # match the values on y axes
        ylim = [0, 1.2]
        yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        ax1.set_ylim(ylim)
        ax1.set_yticks(yticks)
        ax2.set_ylim(ylim)
        ax2.set_yticks(yticks)

    def plot_everything(self, y_prob: list, y_pred: list, y_true: list):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
        fig.subplots_adjust(wspace=0.3)
        self.plot_confusion_matrices(y_pred, y_true, axs=[ax1, ax2])
        self.plot_precision_recall_curve(y_prob, y_true, ax1=ax3)
        self.plot_roc_curve(y_prob, y_true, ax1=ax4)


class GPUSelector:
    """ Code from https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused """
    # Nvidia-smi GPU memory parsing.
    # Tested on nvidia-smi 370.23

    def __init__(self, config: Config):
        self.config = config

    def run_command(self, cmd):
        """Run command, return output as string."""
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
        return output.decode("ascii")

    def list_available_gpus(self):
        """Returns list of available GPU ids."""
        output = self.run_command("nvidia-smi -L")
        # lines of the form GPU 0: TITAN X
        gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
        result = []
        for line in output.strip().split("\n"):
            m = gpu_regex.match(line)
            assert m, "Couldnt parse " + line
            result.append(int(m.group("gpu_id")))
        return result

    def gpu_memory_map(self):
        """Returns map of GPU id to memory allocated on that GPU."""

        output = self.run_command("nvidia-smi")
        gpu_output = output[output.find("GPU Memory"):]
        # lines of the form
        # |    0      8734    C   python                                       11705MiB |
        memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
        rows = gpu_output.split("\n")
        result = {gpu_id: 0 for gpu_id in self.list_available_gpus()}
        for row in gpu_output.split("\n"):
            m = memory_regex.search(row)
            if not m:
                continue
            gpu_id = int(m.group("gpu_id"))
            gpu_memory = int(m.group("gpu_memory"))
            result[gpu_id] += gpu_memory
        return result

    def pick_gpu_lowest_memory(self):
        """Returns GPU with the least allocated memory"""

        print("Selcting optimal GPU")
        best_gpu = ""
        if self.config.PICK_BEST_GPU is True:
            memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in self.gpu_memory_map().items()]
            best_memory, best_gpu = sorted(memory_gpu_map)[0]
            print(f"Using CUDA_VISIBLE_DEVICES {best_gpu}")
        else:
            print("Not using optimal GPU. To enable, set config.PICK_BEST_GPU to True.")

        return best_gpu
