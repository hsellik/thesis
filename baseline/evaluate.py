import wandb
import pickle

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from config import Config
from plots import Plots


def load_data(config: Config, path: str):
    examples = open(path).readlines()
    label_dict = {
        config.BUG_STRING: config.BUG_VALUE,
        config.NO_BUG_STRING: config.NO_BUG_VALUE
    }
    corpus = []
    y = []

    for example in examples:
        corpus.append(example.split(" ", 1)[1])
        y.append(label_dict.get(example.split(" ", 1)[0]))
    return corpus, y


def calculate_metrics(y_true, y_pred, config: Config):
    precision = precision_score(y_true, y_pred, average='binary', pos_label=config.BUG_VALUE)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=config.BUG_VALUE)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=config.BUG_VALUE)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


if __name__ == '__main__':
    config = Config()
    wandb.init(config=config, project="msc_thesis_hendrig")
    print("Adding examples to corpus...")
    X_test, y_test = load_data(config, config.TEST_SET_LOCATION)

    print(f"Number of examples: {len(X_test)}")
    print(f"Number of positive examples: {y_test.count(1)}")
    print(f"Number of negative examples: {y_test.count(0)}")

    # load the vectorizer
    print(f"Loading vectorizer from {config.VECTORIZER_PATH}")
    with open(config.VECTORIZER_PATH, 'rb') as fid:
        vectorizer = pickle.load(fid)

    print(f"Number of features: {len(vectorizer.get_feature_names())}")

    X_test_vectorized = vectorizer.transform(X_test).toarray()

    print("----")
    print(f"X_test_vectorized type: {type(X_test_vectorized)}")
    print(f"X_test_vectorized shape: {X_test_vectorized.shape}")

    print("----")
    print(f"y_test type: {type(y_test)}")
    print(f"y_test shape: {len(y_test)}")

    # Load the model
    print(f"Loading classifier from {config.MODEL_PATH}")
    with open(config.MODEL_PATH, 'rb') as fid:
        clf = pickle.load(fid)

    print("Generating confusion matrix")
    y_pred = clf.predict(X_test_vectorized)
    plots = Plots(config=config)
    plots.plot_confusion_matrices(y_pred=y_pred, y_true=y_test)

    print("Calculating metrics")
    precision, recall, f1, accuracy = calculate_metrics(y_pred=y_pred, y_true=y_test, config=config)
    print(f"precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {accuracy}")
    # Record to WanDB
    print("Logging to wandb")
    wandb.log({'val_f1': f1,
               'val_recall': recall, 'val_precision': precision,
               'val_acc': accuracy, "plots": plt})
    # Close plot
    plt.close()

    print("Done!")
