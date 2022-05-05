import os
import sys
import yaml
import shutil
import logging
import argparse
from pathlib import Path 

from gensim.matutils import Sparse2Corpus

from lda import LdaMalletWithBeta
from utils import load_sparse, load_json

logger = logging.getLogger(__name__)

# such is life--we need specific Mallet version for inference
os.environ["PATH"] = "/workspace/java-bins/jdk-18/bin/:"+ ":".join(os.environ["PATH"].split(":")[1:])
os.environ["JAVA_HOME"] = "/workspace/java-bins/jdk-18/bin"
PATH_TO_MALLET_BINARY = "/workspace/java-bins/Mallet-202108/bin/mallet"

def load_yaml(path):
    with open(path, "r") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_dir")
    parser.add_argument("--original_train_data_file")
    parser.add_argument("--original_vocab_file")
    parser.add_argument("--inference_data_file")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    config = load_yaml(Path(args.model_save_dir, "config.yml"))
    original_train_data_file = Path(config["input_dir"], config["train_path"])
    original_vocab_file = Path(config["input_dir"], config["vocab_path"])

    # load data
    x_train = load_sparse(original_train_data_file)
    x_eval = load_sparse(args.inference_data_file)

    x_train = Sparse2Corpus(x_train, documents_columns=False)
    x_eval = Sparse2Corpus(x_eval, documents_columns=False)

    vocab = load_json(original_vocab_file)
    inv_vocab = dict(zip(vocab.values(), vocab.keys()))

    # make the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.model_save_dir, output_dir, dirs_exist_ok=True)

    # initialize the model
    logger.info("OK")
    lda = LdaMalletWithBeta(
        mallet_path=PATH_TO_MALLET_BINARY,
        num_topics=config["num_topics"],
        id2word=inv_vocab,
        prefix=str(args.output_dir) + "/",
        alpha=config["alpha"],
        beta=config["beta"],
        optimize_interval=config["optimize_interval"],
        topic_threshold=0.,
        iterations=config["iterations"],
        workers=config["workers"],
        random_seed=42,
    )

    # convert the train input (this happened at train time but it may have been deleted)
    lda.convert_input(x_train)
    # now do inference (the estimated document-topic dist will be saved automatically)
    est_topics = lda.__getitem__(x_eval, iterations=100) # default is 100
