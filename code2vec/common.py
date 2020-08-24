import re
import subprocess

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import takewhile, repeat
from typing import List, Optional, Tuple, Iterable
from datetime import datetime
from collections import OrderedDict

from matplotlib.axes import SubplotBase
from sklearn.metrics import auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

from config import Config


class common:

    @staticmethod
    def normalize_word(word):
        stripped = re.sub(r'[^a-zA-Z]', '', word)
        if len(stripped) == 0:
            return word.lower()
        else:
            return stripped.lower()

    @staticmethod
    def _load_vocab_from_histogram(path, min_count=0, start_from=0, return_counts=False):
        with open(path, 'r') as file:
            word_to_index = {}
            index_to_word = {}
            word_to_count = {}
            next_index = start_from
            for line in file:
                line_values = line.rstrip().split(' ')
                if len(line_values) != 2:
                    continue
                word = line_values[0]
                count = int(line_values[1])
                if count < min_count:
                    continue
                if word in word_to_index:
                    continue
                word_to_index[word] = next_index
                index_to_word[next_index] = word
                word_to_count[word] = count
                next_index += 1
        result = word_to_index, index_to_word, next_index - start_from
        if return_counts:
            result = (*result, word_to_count)
        return result

    @staticmethod
    def load_vocab_from_histogram(path, min_count=0, start_from=0, max_size=None, return_counts=False):
        if max_size is not None:
            word_to_index, index_to_word, next_index, word_to_count = \
                common._load_vocab_from_histogram(path, min_count, start_from, return_counts=True)
            if next_index <= max_size:
                results = (word_to_index, index_to_word, next_index)
                if return_counts:
                    results = (*results, word_to_count)
                return results
            # Take min_count to be one plus the count of the max_size'th word
            min_count = sorted(word_to_count.values(), reverse=True)[max_size] + 1
        return common._load_vocab_from_histogram(path, min_count, start_from, return_counts)

    @staticmethod
    def load_json(json_file):
        data = []
        with open(json_file, 'r') as file:
            for line in file:
                current_program = common.process_single_json_line(line)
                if current_program is None:
                    continue
                for element, scope in current_program.items():
                    data.append((element, scope))
        return data

    @staticmethod
    def load_json_streaming(json_file):
        with open(json_file, 'r') as file:
            for line in file:
                current_program = common.process_single_json_line(line)
                if current_program is None:
                    continue
                for element, scope in current_program.items():
                    yield (element, scope)

    @staticmethod
    def save_word2vec_file(output_file, index_to_word, vocab_embedding_matrix: np.ndarray):
        assert len(vocab_embedding_matrix.shape) == 2
        vocab_size, embedding_dimension = vocab_embedding_matrix.shape
        output_file.write('%d %d\n' % (vocab_size, embedding_dimension))
        for word_idx in range(0, vocab_size):
            assert word_idx in index_to_word
            word_str = index_to_word[word_idx]
            output_file.write(word_str + ' ')
            output_file.write(' '.join(map(str, vocab_embedding_matrix[word_idx])) + '\n')

    @staticmethod
    def calculate_max_contexts(file):
        contexts_per_word = common.process_test_input(file)
        return max(
            [max(l, default=0) for l in [[len(contexts) for contexts in prog.values()] for prog in contexts_per_word]],
            default=0)

    @staticmethod
    def binary_to_string(binary_string):
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list):
        return [common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(binary_string_matrix):
        return [common.binary_to_string_list(l) for l in binary_string_matrix]

    @staticmethod
    def load_file_lines(path):
        with open(path, 'r') as f:
            return f.read().splitlines()

    @staticmethod
    def split_to_batches(data_lines, batch_size):
        for x in range(0, len(data_lines), batch_size):
            yield data_lines[x:x + batch_size]

    @staticmethod
    def legal_method_names_checker(special_words, name):
        return name != special_words.OOV and re.match(r'^[a-zA-Z|]+$', name)

    @staticmethod
    def filter_impossible_names(special_words, top_words):
        result = list(filter(lambda word: common.legal_method_names_checker(special_words, word), top_words))
        return result

    @staticmethod
    def get_subtokens(str):
        return str.split('|')

    @staticmethod
    def parse_prediction_results(raw_prediction_results, unhash_dict, special_words, topk: int = 5) -> List['MethodPredictionResults']:
        prediction_results = []
        for single_method_prediction in raw_prediction_results:
            current_method_prediction_results = MethodPredictionResults(single_method_prediction.original_name)
            current_method_prediction_results.append_probabilities(single_method_prediction.probs)
            for i, predicted in enumerate(single_method_prediction.topk_predicted_words):
                if predicted == special_words.OOV:
                    continue
                suggestion_subtokens = common.get_subtokens(predicted)
                current_method_prediction_results.append_prediction(
                    suggestion_subtokens, single_method_prediction.topk_predicted_words_scores[i].item())
            topk_attention_per_context = [
                (key, single_method_prediction.attention_per_context[key])
                for key in sorted(single_method_prediction.attention_per_context,
                                  key=single_method_prediction.attention_per_context.get, reverse=True)
            ][:topk]
            for context, attention in topk_attention_per_context:
                token1, hashed_path, token2 = context
                if hashed_path in unhash_dict:
                    unhashed_path = unhash_dict[hashed_path]
                    current_method_prediction_results.append_attention_path(attention.item(), token1=token1,
                                                                            path=unhashed_path, token2=token2)
            prediction_results.append(current_method_prediction_results)
        return prediction_results

    @staticmethod
    def tf_get_first_true(bool_tensor: tf.Tensor) -> tf.Tensor:
        bool_tensor_as_int32 = tf.cast(bool_tensor, dtype=tf.int32)
        cumsum = tf.cumsum(bool_tensor_as_int32, axis=-1, exclusive=False)
        return tf.logical_and(tf.equal(cumsum, 1), bool_tensor)

    @staticmethod
    def count_lines_in_file(file_path: str):
        with open(file_path, 'rb') as f:
            bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
            return sum(buf.count(b'\n') for buf in bufgen)

    @staticmethod
    def squeeze_single_batch_dimension_for_np_arrays(arrays):
        assert all(array is None or isinstance(array, np.ndarray) or isinstance(array, tf.Tensor) for array in arrays)
        return tuple(
            None if array is None else np.squeeze(array, axis=0)
            for array in arrays
        )

    @staticmethod
    def get_first_match_word_from_top_predictions(special_words, original_name, top_predicted_words) -> Optional[Tuple[int, str]]:
        normalized_original_name = common.normalize_word(original_name)
        for suggestion_idx, predicted_word in enumerate(common.filter_impossible_names(special_words, top_predicted_words)):
            normalized_possible_suggestion = common.normalize_word(predicted_word)
            if normalized_original_name == normalized_possible_suggestion:
                return suggestion_idx, predicted_word
        return None

    @staticmethod
    def now_str():
        return datetime.now().strftime("%Y%m%d-%H%M%S: ")

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def get_unique_list(lst: Iterable) -> list:
        return list(OrderedDict(((item, 0) for item in lst)).keys())


class MethodPredictionResults:
    def __init__(self, original_name):
        self.original_name = original_name
        self.probs = list()
        self.predictions = list()
        self.attention_paths = list()

    def append_prediction(self, name, probability):
        self.predictions.append({'name': name, 'probability': probability})

    def append_probabilities(self, probs):
        self.probs = probs

    def append_attention_path(self, attention_score, token1, path, token2):
        self.attention_paths.append({'score': attention_score,
                                     'path': path,
                                     'token1': token1,
                                     'token2': token2})


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

        best_gpu = ""
        if self.config.PICK_BEST_GPU is True:
            memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in self.gpu_memory_map().items()]
            best_memory, best_gpu = sorted(memory_gpu_map)[0]
            print(f"Using CUDA_VISIBLE_DEVICES {best_gpu}")
        else:
            print("Not using optimal GPU. To enable, set config.PICK_BEST_GPU to True.")

        return best_gpu
