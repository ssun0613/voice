import argparse

import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    config = yaml.load(open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
