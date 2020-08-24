import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from common import Plots
from config import Config
from model import Model

import wandb

class Evaluator:

    def __init__(self, config: Config, model: Model):
        self.model = model
        self.config = config

    def evaluate(self):
        test_accuracy, test_precision, test_recall, test_f1, test_y_true, test_y_pred, test_y_prob = self.model.evaluate()
        self.print_statistics(test_accuracy, test_precision, test_recall, test_f1, test_y_true, test_y_pred)
        self.log_results_to_wandb(test_accuracy, test_f1, test_precision, test_recall, test_y_prob, test_y_pred, test_y_true)

    def print_statistics(self, accuracy, precision: float, recall: float, f1: float, y_true: list, y_pred: list):
        print('Accuracy: ' + str(accuracy) + 'Precision: ' + str(precision) + ', Recall: ' + str(recall) + ', F1: ' + str(f1))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[self.config.NO_BUG_STRING, self.config.BUG_STRING]).ravel()
        print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
        print("True label statistics")
        print(f"No. of bug: {y_true.count(self.config.BUG_STRING)}")
        print(f"No. of nobug: {y_true.count(self.config.NO_BUG_STRING)}")
        print(y_true)
        print("Predicted label statistics")
        print(f"No. of bug: {y_pred.count('bug')}")
        print(f"No. of nobug: {y_pred.count('nobug')}")
        print(y_pred)

    def log_results_to_wandb(self, accuracy: float, f1: float, precision: float, recall: float, y_prob: list, y_pred: list, y_true: list):
        # Create confusion matrices
        if not y_true:
            # Ensure that at least something gets plotted, with
            y_true = ["nobug"]
            y_pred = ["nobug"]
            y_prob = [1.0]
        print("Creating plots for evaluation and logging to wandb...")
        plots = Plots(self.config)
        plots.plot_everything(y_prob=y_prob, y_pred=y_pred, y_true=y_true)
        # Record to WanDB
        wandb.log({'val_f1': f1,
                   'val_recall': recall, 'val_precision': precision,
                   'val_acc': accuracy, "plots": plt})
        # Close plot
        plt.close()
        print("Finished")
