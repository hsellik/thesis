import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from config import Config


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
            cm = confusion_matrix(y_true, y_pred, labels=[self.config.BUG_VALUE, self.config.NO_BUG_VALUE],
                                  normalize=normalize_option)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=[self.config.BUG_STRING, self.config.NO_BUG_STRING])
            disp.plot(include_values=True, cmap='Blues', ax=axs[index], values_format=values_format,
                      xticks_rotation='horizontal')
            axs[index].title.set_text(plot_name)
            index += 1
