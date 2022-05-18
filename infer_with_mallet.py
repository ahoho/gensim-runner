import os
import sys
import yaml
import shutil
import logging
import argparse
from pathlib import Path 

import numpy as np
from gensim.matutils import Sparse2Corpus

from lda import LdaMalletWithBeta
from utils import load_sparse, load_json

logger = logging.getLogger(__name__)

# such is life--we need specific Mallet version for inference
os.environ["PATH"] = "/workspace/java-bins/jdk-18/bin/:"+ ":".join(os.environ["PATH"].split(":")[1:])
os.environ["JAVA_HOME"] = "/workspace/java-bins/jdk-18/bin"
# TODO: move these into shell scripts

def load_yaml(path):
    with open(path, "r") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def load_mallet_doc_topic(fpath):
    with open(fpath) as infile:
        return np.array([
            [float(x) for x in line.strip().split("\t")[2:]]
            for line in infile
        ])


def retrieve_estimates(model_dir, eval_data=None, mallet_path=None, **kwargs):
    """
    Loads the mallet model and the topic-word distribution,
    then instantiates the encoder portion and does a forward pass to get the
    training-set document-topic estimates

    If `eval_data` is provided, will infer new document-topic estimates for the data
    """
    model_dir = Path(model_dir)
    config = load_yaml(model_dir / "config.yml")

    # train estimates will have already been saved, so this terminates early
    if eval_data is None:
        # Load the text document-topic estimate as a numpy matrix
        topic_word = np.load(model_dir / "beta.npy")
        doc_topic = load_mallet_doc_topic(model_dir / "doctopics.txt")
        return topic_word, doc_topic
    
    # NOTE: there is a some hacky moving-about of files because of 
    # mallet/gensim idiosyncracies. basically the logic is:
    # have mallet make the necessary train corpus files if they do not already exist,
    # otherwise copy them from the data directory
    mallet_input_dir = Path(config["input_dir"], "mallet")

    original_vocab_file = Path(config["input_dir"], config["vocab_path"])
    vocab = load_json(original_vocab_file)
    inv_vocab = dict(zip(vocab.values(), vocab.keys()))

    # initialize the model
    lda = LdaMalletWithBeta(
        mallet_path=mallet_path or config["mallet_path"],
        num_topics=config["num_topics"],
        id2word=inv_vocab,
        prefix=str(model_dir) + "/",
        alpha=config["alpha"],
        beta=config["beta"],
        optimize_interval=config["optimize_interval"],
        topic_threshold=0.,
        iterations=config["iterations"],
        workers=config["workers"],
        random_seed=42,
    )

    # need to use the previous corpus to create the new one, and it has to live
    # in the same directory as our model
    if (mallet_input_dir / "corpus.mallet").exists():
        # if exists, copy it to the saved model directory for infrence
        shutil.copyfile(mallet_input_dir / "corpus.mallet", model_dir / "corpus.mallet")
    else:
        # otherwise, convert the standard dtm file and make a permanent copy in the input directory
        original_train_data_file = Path(config["input_dir"], config["train_path"])
        train_data = load_sparse(original_train_data_file)
        train_data = Sparse2Corpus(train_data, documents_columns=False)
        lda.convert_input(train_data)

        mallet_input_dir.mkdir()
        shutil.copyfile(model_dir / "corpus.mallet", mallet_input_dir / "corpus.mallet")
    
    # now do inference (the estimated document-topic dist will be saved automatically)
    eval_data = Sparse2Corpus(eval_data, documents_columns=False)
    doc_topic_gensim = lda.__getitem__(eval_data, iterations=100) # default is 100

    # convert to numpy
    doc_topic = np.zeros((len(doc_topic_gensim), config["num_topics"]), dtype=np.float32)
    for doc_idx, doc in enumerate(doc_topic_gensim):
        for topic_idx, prob in doc:
            doc_topic[doc_idx, topic_idx] = prob
    
    # finally, remove the unnecessary files
    (model_dir / "corpus.txt").unlink()
    (model_dir / "corpus.mallet").unlink()
    (model_dir / "corpus.mallet.infer").unlink()

    return doc_topic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--inference_data_file")
    parser.add_argument("--mallet_path", default=None)
    parser.add_argument("--output_fpath") # this is ignored
    args = parser.parse_args()

    assert Path(args.model_dir, "inferencer.mallet").exists(), f"Model does not exist at {args.model_dir}/inferencer.mallet"

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    eval_data = load_sparse(args.inference_data_file)
    doc_topic = retrieve_estimates(
        model_dir=args.model_dir,
        eval_data=eval_data,
        mallet_path=args.mallet_path,
    )
    
    if args.output_fpath.endswith(".txt"):
        shutil.move(Path(args.model_dir, "doctopics.txt.infer"), args.output_fpath)
    else:
        np.save(args.output_fpath, doc_topic)
        Path(args.model_dir, "doctopics.txt.infer").unlink()
