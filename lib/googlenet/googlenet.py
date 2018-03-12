import tensorflow as tf
import numpy as np
from lib.networks.base_network import Net


class GoogleNet(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)

    def init_saver(self):
        pass

    def build_model(self):
        pass
