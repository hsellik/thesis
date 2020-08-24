import math
import wandb
import pickle

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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
    X_train, y_train = load_data(config, config.TRAIN_SET_LOCATION)
    X_test, y_test = load_data(config, config.TEST_SET_LOCATION)

    print(f"Number of examples: {len(X_train) + len(X_test)}")
    print(f"Number of positive examples: {y_train.count(1) + y_test.count(1)}")
    print(f"Number of negative examples: {y_train.count(0) + y_test.count(0)}")

    print("Vectorizing training and test sets...")
    max_features = math.floor((len(X_train)) / config.NUM_EXAMPLES_PER_FEATURE)
    vectorizer = TfidfVectorizer(max_features=max_features)

    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
    X_test_vectorized = vectorizer.transform(X_test).toarray()

    # Save the vectorizer
    print(f"Saving vectorizer to {config.VECTORIZER_PATH}")
    with open(config.VECTORIZER_PATH, 'wb') as fid:
       pickle.dump(vectorizer, fid)
    print(f"Number of features: {len(vectorizer.get_feature_names())}")

    print("----")
    print(f"X_train_vectorized type: {type(X_train_vectorized)}")
    print(f"X_train_vectorized shape: {X_train_vectorized.shape}")

    print("----")
    print(f"y_train type: {type(y_train)}")
    print(f"y_train shape: {len(y_train)}")

    print("----")
    print(f"X_test_vectorized type: {type(X_test_vectorized)}")
    print(f"X_test_vectorized shape: {X_test_vectorized.shape}")

    print("----")
    print(f"y_test type: {type(y_test)}")
    print(f"y_test shape: {len(y_test)}")

    print("Fitting classifer")
    # For Random Forest, the more trees the better?
    # https://stats.stackexchange.com/a/348246
    clf = RandomForestClassifier(n_estimators=config.N_TREES, n_jobs=config.N_JOBS)
    clf.fit(X_train_vectorized, y_train)

    # Save the model
    print(f"Saving classifier to {config.MODEL_PATH}")
    with open(config.MODEL_PATH, 'wb') as fid:
       pickle.dump(clf, fid)

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
