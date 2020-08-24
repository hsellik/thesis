class Config:
    @staticmethod
    def get_default_config(args):
        config = Config(args)
        config.GRAM_SIZE = 3
        config.SHORTER_SEQUENCE_LEN = 3
        config.LONGER_SEQUENCE_LEN = 5
        config.REPORTING_SIZE = 500
        config.MINIMUM_TOKEN_OCCURENCE = 3
        # Make sure that the temp data of another project doesn't interfere with process
        if len(config.DATA_PATH) > 10:
            config.TEMP_DATA_PATH = "temp_filtered_tokens_" + config.DATA_PATH[-10:] + ".txt"
        else:
            config.TEMP_DATA_PATH = "temp_filtered_tokens_" + config.DATA_PATH + ".txt"
        config.METHOD_TOKEN_SPLITTER = " "
        config.TOKEN_SPLITTER = ","
        config.METHOD_INDEX = 0
        config.TOKEN_INDEX = 1

        return config

    def __init__(self, args):
        self.GRAM_SIZE = 0
        self.SHORTER_SEQUENCE_LEN = 0
        self.LONGER_SEQUENCE_LEN = 0
        self.REPORTING_SIZE = 0
        self.MINIMUM_TOKEN_OCCURENCE = 0
        self.DATA_PATH = args.data_path if args.data_path is not None else ''
        self.SAVE_PATH = args.save_path if args.save_path is not None else ''
        self.LOAD_PATH = args.load_path if args.load_path is not None else ''
        self.TEMP_DATA_PATH = ""
        self.METHOD_TOKEN_SPLITTER = " "
        self.TOKEN_SPLITTER = ","
        self.METHOD_INDEX = 0
        self.TOKEN_INDEX = 1
