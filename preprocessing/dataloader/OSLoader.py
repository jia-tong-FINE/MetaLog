import sys

sys.path.extend([".",".."])
from CONSTANTS import *
from collections import OrderedDict
from preprocessing.BasicLoader import BasicDataLoader


class OSLoader(BasicDataLoader):
    def __init__(self, in_file=None, ab_in_file=None,
                 window_size=20,
                 dataset_base=os.path.join(PROJECT_ROOT, 'datasets/OpenStack'),
                 semantic_repr_func=None):
        super(OSLoader, self).__init__()

        # Construct logger.
        self.logger = logging.getLogger('OSLoader')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'OSLoader.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.info(
            'Construct self.logger success, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))

        if not os.path.exists(in_file):
            self.logger.error('Input file not found, please check.')
            exit(1)
        self.in_file = in_file
        self.ab_in_file = ab_in_file
        self.window_size = window_size
        self.dataset_base = dataset_base
        self.log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
        # self.remove_cols = None
        self._load_raw_log_seqs()
        self.semantic_repr_func = semantic_repr_func
        pass

    def logger(self):
        return self.logger
    
    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
    
    def logger(self):
        return self.logger

    def _pre_process(self, line):
        headers, regex = self.generate_logformat_regex(self.log_format)
        match = regex.search(line.strip())
        if match != None:
            message = [match.group(header) for header in headers]
            return message[-1]
        else:
            return ''

    def _load_raw_log_seqs(self):
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        label_file = os.path.join(self.dataset_base, 'label.txt')
        if os.path.exists(sequence_file) and os.path.exists(label_file):
            self.logger.info('Start load from previous extraction. File path %s' % sequence_file)
            with open(sequence_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(':')
                    block = tokens[0]
                    seq = tokens[1].split()
                    if block not in self.block2seqs.keys():
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(x) for x in seq]
            with open(label_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    block_id, label = line.strip().split(':')
                    self.block2label[block_id] = label

        else:
            self.logger.info('Start loading OpenStack log sequences.')
            # read normal lines
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                block_idx = 0
                num = 0
                log_id = 0
                for line in lines:
                    if num < self.window_size:
                        if num == 0:
                            self.block2seqs[str(block_idx)] = [log_id]
                        else:
                            self.block2seqs[str(block_idx)].append(log_id)
                        num += 1
                        log_id += 1
                    else:
                        self.blocks.append(str(block_idx))
                        label = 'Normal'
                        self.block2label[str(block_idx)] = label
                        # self.block2seqs[str(block_idx)].append(log_id)
                        block_idx += 1
                        num = 0
                        

            # read abnormal lines
            with open(self.ab_in_file, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                num = 0
                for line in lines:
                    if num < self.window_size:
                        if num == 0:
                            self.block2seqs[str(block_idx)] = [log_id]
                        else:
                            self.block2seqs[str(block_idx)].append(log_id)
                        num += 1
                        log_id += 1
                    else:
                        self.blocks.append(str(block_idx))
                        label = 'Anomalous'
                        self.block2label[str(block_idx)] = label
                        # self.block2seqs[str(block_idx)].append(log_id)
                        block_idx += 1
                        num = 0
                        

            with open(sequence_file, 'w', encoding='utf-8') as writer:
                for block in self.blocks:
                    writer.write(':'.join([block, ' '.join([str(x) for x in self.block2seqs[block]])]) + '\n')

            with open(label_file, 'w', encoding='utf-8') as writer:
                for block in self.block2label.keys():
                    writer.write(':'.join([block, self.block2label[block]]) + '\n')

        self.logger.info('Extraction finished successfully.')
        pass


if __name__ == '__main__':
    from representations.templates.statistics import Simple_template_TF_IDF

    semantic_encoder = Simple_template_TF_IDF()
    loader = OSLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/OpenStack/openstack_normal1.log'),
                      ab_in_file=os.path.join(PROJECT_ROOT, 'datasets/OpenStack/openstack_abnormal.log'),
                       dataset_base=os.path.join(PROJECT_ROOT, 'datasets/OpenStack'),
                       semantic_repr_func=semantic_encoder.present)
    loader.parse_by_IBM(config_file=os.path.join(PROJECT_ROOT, 'conf/OpenStack.ini'),
                        persistence_folder=os.path.join(PROJECT_ROOT, 'datasets/OpenStack/persistences'))
    pass
