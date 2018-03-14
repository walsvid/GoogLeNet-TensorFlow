import yaml


class Config(object):
    def __init__(self, cfg_):
        """
        :type cfg_: dict
        :param cfg_:
        """
        self.config = cfg_


class TrainNetConfig(Config):
    def __init__(self, cfg_):
        """
        :type cfg_: dict
        :param cfg_:
        """
        super().__init__(cfg_)
        self.checkpoint_dir = self.config['CHECKPOINT_DIR']
        self.max_to_keep = self.config['MAX_TO_KEEP']
        self.n_classes = self.config['N_CLASSES']
        self.learning_rate = self.config['LEARNING_RATE']
        self.max_step = self.config['MAX_STEP']
        self.is_pretrain = self.config['IS_PRETRAIN']
        self.batch_size = self.config['BATCH_SIZE']
        self.image_width = self.config['IMAGE_SIZE']['WIDTH']
        self.image_height = self.config['IMAGE_SIZE']['HEIGHT']
        self.image_depth = self.config['IMAGE_SIZE']['DEPTH']
        self.pre_train_weight = self.config['PRE_TRAIN_WEIGHT']


class TestNetConfig(Config):
    def __init__(self, cfg_):
        super().__init__(cfg_)


class DataConfig(Config):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        self.image_width = self.config['IMAGE_SIZE']['WIDTH']
        self.image_height = self.config['IMAGE_SIZE']['HEIGHT']
        self.image_depth = self.config['IMAGE_SIZE']['DEPTH']
        self.data_dir = self.config['DATA_DIR']
        self.batch_size = self.config['BATCH_SIZE']
        self.n_classes = self.config['N_CLASSES']


class ConfigReader(object):
    def __init__(self, yaml_file_):
        with open(yaml_file_) as yf:
            self.total_config = yaml.load(yf)

    def get_train_config(self):
        try:
            return {**self.total_config['TRAIN'], **self.total_config['DATA']}
        except KeyError:
            return None

    def get_test_config(self):
        try:
            return {**self.total_config['TEST'], **self.total_config['DATA']}
        except KeyError:
            return None