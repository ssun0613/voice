import argparse

import yaml

from preprocessor import ljspeech


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)


if __name__ == "__main__":

    config = yaml.load(open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    main(config)
