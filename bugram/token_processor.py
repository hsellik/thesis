from collections import Counter

from config import Config


class TokenProcessor:

    def __init__(self, config: Config):
        self.minimum_token_occurence = config.MINIMUM_TOKEN_OCCURENCE
        self.method_token_splitter = config.METHOD_TOKEN_SPLITTER
        self.token_splitter = config.TOKEN_SPLITTER
        self.method_index = config.METHOD_INDEX
        self.token_index = config.TOKEN_INDEX
        self.data = config.DATA_PATH
        self.temp_data = config.TEMP_DATA_PATH
        self.sequence_length = config.SHORTER_SEQUENCE_LEN

    def remove_rare_tokens(self):
        """ Remove tokens occuring more than config.MINIMUM_TOKEN_OCCURENCE and write results to a new temp file """
        tokens = []
        with open(self.data) as r:
            for line in r:
                line = line.rstrip()
                if len(line.split(" ")) == 2:
                    tokens.extend(line.split(" ")[self.token_index].split(","))
        counted_tokens = Counter(tokens)

        with open(self.temp_data, "w") as w, open(self.data, "r") as r:
            for line in r:
                line = line.rstrip()
                if len(line.split(" ")) == 2:
                    method_name = line.split(self.method_token_splitter)[self.method_index]
                    tokens_to_add = line.split(self.method_token_splitter)[self.token_index].split(self.token_splitter)
                    # Only add tokens that occur over config.minimum_token_occurence
                    filtered_tokens_to_add = []
                    for token in tokens_to_add:
                        if counted_tokens[token] >= self.minimum_token_occurence:
                            filtered_tokens_to_add.append(token)
                    if len(filtered_tokens_to_add) > 0:
                        w.write(f"{method_name}{self.method_token_splitter}{self.token_splitter.join(filtered_tokens_to_add)}\n")

    def get_token_sequences(self):
        """ Presumes infrequent tokens are filtered and results saved in config.TEMP_DATA_PATH

            :returns a list of tokens for each method as [[token1, token2, ..., token], [token1, token2, ..., token]] """
        token_sequences = []
        with open(self.temp_data) as r:
            for line in r:
                line = line.rstrip()
                if len(line.split(" ")) == 2:
                    token_sequences.append(line.split(self.method_token_splitter)[self.token_index].split(self.token_splitter))
        return token_sequences

    def get_token_sequences_for_evaluation(self):
        """ Return token sequences as [(method_name1 token1, token2, ... token), (method_name2 token1, ... token)] """
        file_token_sequences = []
        with open(self.temp_data) as r:
            for line in r:
                file_token_sequences.append(line.rstrip())

        return file_token_sequences

    @staticmethod
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
