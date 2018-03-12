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


class TestNetConfig(Config):
    def __init__(self, cfg_):
        super().__init__(cfg_)


class DataConfig(Config):
    def __init__(self, cfg_):
        super().__init__(cfg_)


class ConfigReader(object):
    def __init__(self, yaml_file_):
        with open(yaml_file_) as yf:
            self.total_config = yaml.load(yf)
