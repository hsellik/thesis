class Config:

    def __init__(self):
        self.NUM_EXAMPLES_PER_FEATURE = 300
        self.BUG_STRING = "bug"
        self.NO_BUG_STRING = "nobug"
        self.BUG_VALUE = 1
        self.NO_BUG_VALUE = 0
        self.TRAIN_SET_LOCATION = "./data/tokens_balanced_train.txt"
        self.TEST_SET_LOCATION = "./data/tokens_balanced_test.txt"
        self.VECTORIZER_PATH = "./tfidf_balanced_vectorizer.pkl"
        self.MODEL_PATH = "./rf_balanced_classifier.pkl"
        self.N_JOBS = 20
        self.N_TREES = 75
