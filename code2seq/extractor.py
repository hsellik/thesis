import subprocess

from common import PathContextInformation


class Extractor:
    def __init__(self, config, jar_path, max_path_length, max_path_width):
        self.config = config
        self.max_path_length = max_path_length
        self.max_path_width = max_path_width
        self.bad_characters_table = str.maketrans('', '', '\t\r\n')
        self.jar_path = jar_path

    def process_line(self, line):
        pc_info_dict = {}
        result = []

        line = line.rstrip()
        method_name = line.split(' ')[0]
        current_result_line_parts = [method_name]
        contexts = line.split(' ')[1:]
        contexts_processed = 0
        for context in contexts[:self.config.DATA_NUM_CONTEXTS]:
            if len(context.split(',')) == 3:
                context_dict = {}
                context_dict["name1"] = context.split(',')[0]
                context_dict["shortPath"] = context.split(',')[1]
                context_dict["name2"] = context.split(',')[2]
                pc_info = PathContextInformation(context_dict)
                current_result_line_parts += [str(pc_info)]
                pc_info_dict[(pc_info.token1, pc_info.shortPath, pc_info.token2)] = pc_info
                contexts_processed += 1
        space_padding = ' ' * (self.config.DATA_NUM_CONTEXTS - contexts_processed)
        result_line = ' '.join(current_result_line_parts) + space_padding
        result.append(result_line)

        return result, pc_info_dict

    def extract_paths(self, path):
        command = ['java', '-cp', self.jar_path, 'JavaExtractor.App', '--code2seq', 'true', '--max_path_length',
                   str(self.max_path_length), '--max_path_width', str(self.max_path_width), '--file', path,
                   '--evaluate', 'true', '--off_by_one', 'true']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if len(out) == 0:
            err = err.decode()
            raise ValueError(err)
        output = out.decode().splitlines()
        # output = out.decode().splitlines()
        # predict_lines = ['get|name string,Cls0|Mth|Nm1,METHOD_NAME string,Cls0|Mth|Bk|Ret|Nm0,name METHOD_NAME,Nm1|Mth|Bk|Ret|Nm0,name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ']
        # pc_info_dict = {('string', 'Cls0|Mth|Nm1', 'METHOD_NAME'), ('string', 'Cls0|Mth|Bk|Ret|Nm0', 'name'), ('METHOD_NAME', 'Nm1|Mth|Bk|Ret|Nm0', 'name')}
        pc_info_dict = {}
        result = []
        for line in output:
            method_name = line.split(' ')[0]
            current_result_line_parts = [method_name]
            contexts = line.split(' ')[1:]
            for context in contexts[:self.config.DATA_NUM_CONTEXTS]:
                context_dict = {}
                context_dict["name1"] = context.split(',')[0]
                context_dict["shortPath"] = context.split(',')[1]
                context_dict["name2"] = context.split(',')[2]
                pc_info = PathContextInformation(context_dict)
                current_result_line_parts += [str(pc_info)]
                pc_info_dict[(pc_info.token1, pc_info.shortPath, pc_info.token2)] = pc_info
            space_padding = ' ' * (self.config.DATA_NUM_CONTEXTS - len(contexts))
            result_line = ' '.join(current_result_line_parts) + space_padding
            result.append(result_line)
        return result, pc_info_dict