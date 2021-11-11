import logging
import re
import shutil
import subprocess
from pathlib import Path

import configargparse
import numpy as np
import pandas as pd
from scipy import sparse

from gensim import utils
from gensim.matutils import Sparse2Corpus
from gensim.models.wrappers import LdaMallet
from gensim.models.ldamulticore import LdaMulticore, LdaModel

from utils import NPMI, compute_tu, compute_to, load_sparse, load_json, save_json, save_topics

logger = logging.getLogger(__name__)

class LdaMalletWithBeta(LdaMallet):
    def __init__(self, beta=None, num_top_words=50, *args, **kwargs):
        self.beta = beta
        self.num_top_words = num_top_words

        self.training_log = None
        self.training_topics = None
        self.training_scores = None

        super().__init__(*args, **kwargs)

    def convert_input(self, corpus, infer=False, serialize_corpus=True):
        """Convert corpus to Mallet format and save it to a temporary text file.
        Identical to the original `convert_input` but does not remove stopwords.
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Collection of texts in BoW format.
        infer : bool, optional
            ...
        serialize_corpus : bool, optional
            ...
        """
        if serialize_corpus:
            logger.info("serializing temporary corpus to %s", self.fcorpustxt())
            with utils.open(self.fcorpustxt(), 'wb') as fout:
                self.corpus2mallet(corpus, fout)

        # convert the text file above into MALLET's internal format
        cmd = \
            self.mallet_path + \
            " import-file --preserve-case --keep-sequence " \
            "--token-regex \"\\S+\" --input %s --output %s"
        if infer:
            cmd += ' --use-pipe-from ' + self.fcorpusmallet()
            cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet() + '.infer')
        else:
            cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet())
        logger.info("converting temporary corpus to MALLET format with %s", cmd)
        utils.check_output(args=cmd, shell=True)

    def train(self, corpus):
        self.convert_input(corpus, infer=False)
        args = [
            self.mallet_path,
            "train-topics",
            "--input", f"{self.fcorpusmallet()}",
            "--num-topics", f"{self.num_topics}",
            "--alpha", f"{self.alpha}",
            "--optimize-interval", f"{self.optimize_interval}",

            "--num-threads", f"{self.workers}",
            "--output-state", f"{self.fstate()}",
            "--output-doc-topics", f"{self.fdoctopics()}",
            "--output-topic-keys", f"{self.ftopickeys()}",
            "--num-top-words", f"{self.num_top_words}",

            "--num-iterations", f"{self.iterations}",
            "--inferencer-filename", f"{self.finferencer()}",
            "--doc-topics-threshold", f"{self.topic_threshold}",
            "--random-seed", f"{self.random_seed}"
        ]
        if self.beta is not None:
            args += ["--beta", f"{self.beta}"]
        logger.info("training MALLET LDA with {}".format(" ".join(args)))
        self.training_log = ""
        with subprocess.Popen(args=args, stderr=subprocess.PIPE, bufsize=1, text=True) as p:
            for line in p.stderr:
                print(line, end='')
                self.training_log += line
        self.training_topics = self.parse_log(self.training_log)

        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

    def parse_log(self, training_log):
        training_topics = []
        topics = []
        for line in training_log.split("\n"):
            if re.match("[0-9]", line):
                topic = line.split("\t")[-1]
                topics.append(topic.strip().split(" "))
                if len(topics) == self.num_topics:
                    training_topics.append(topics)
            else: # restart collection
                topics = []
        return training_topics

    def return_optimal_topic_terms(self, score_fn):
        """
        Return the topics with the highest `score_fn` score

        `score_fn` should take a list of word lists as input and return a scalar value
        """
        if self.training_log is None:
            raise ValueError("Model has not been trained!")
        if self.training_topics is None:
            self.training_topics = self.parse_log(self.training_log)
        scores = [score_fn(topics) for topics in self.training_topics]
        return self.training_topics[np.argmax(scores)], np.max(scores)
        

def main(args):
    np.random.seed(args.seed)
    if args.input_dir is not None:
        args.train_path = Path(args.input_dir,  args.train_path)
        args.eval_path = Path(args.input_dir, args.eval_path)
        args.vocab_path = Path(args.input_dir, args.vocab_path)

    x_train = load_sparse(args.train_path)

    if args.save_all_topics:
        Path(args.output_dir, "topics").mkdir(exist_ok=True)

    if not args.eval_path and args.eval_split and args.eval_split > 0:
        split_idx = np.random.choice(
            (True, False),
            size=x_train.shape[0],
            p=(1-args.eval_split, args.eval_split),
        )
        x_train, x_val = x_train[split_idx], x_train[~split_idx]
    else:
        x_val = x_train

    if args.eval_path:
        x_val = load_sparse(args.eval_path)
    x_train = Sparse2Corpus(x_train, documents_columns=False)

    # load the vocabulary
    vocab = None
    if args.vocab_path is not None:
        vocab = load_json(args.vocab_path)
        inv_vocab = {i: v for v, i in vocab.items()}

    if args.model == "gensim":
        extra_kwargs = {}
        lda_class = LdaModel
        if not args.alpha == "auto" and not args.eta == "auto":
            lda_class = LdaMulticore
            extra_kwargs["workers"] = args.workers
        lda = lda_class(
            corpus=x_train,
            num_topics=args.num_topics,
            id2word=inv_vocab,
            alpha=args.alpha,
            eta=args.eta,
            minimum_probability=0.,
            decay=args.decay,
            passes=args.passes,
            iterations=args.iterations,
            random_state=args.seed,
            **extra_kwargs,
        )

    if args.model == "mallet":
        lda = LdaMalletWithBeta(
            mallet_path=args.mallet_path,
            corpus=x_train,
            num_topics=args.num_topics,
            id2word=inv_vocab,
            alpha=args.alpha,
            beta=args.beta,
            num_top_words=args.topic_words_to_save,
            optimize_interval=args.optimize_interval,
            topic_threshold=0.,
            iterations=args.iterations,
            prefix=str(Path(args.output_dir)) + "/",
            workers=args.workers,
            random_seed=args.seed,
        )

    npmi_scorer = NPMI((x_val > 0).astype(int))
    if args.optimize_for_coherence:
        # TODO: currently only implemented for LdaMallet
        # TODO: support other coherence measures
        npmi_score_fn = lambda topics: np.mean(npmi_scorer.compute_npmi(
            topics=topics, vocab=vocab, n=args.eval_words
        ))
        topic_terms, npmi = lda.return_optimal_topic_terms(npmi_score_fn)
        # TODO: full NPMI, not mean
        save_topics(
            topic_terms,
            fpath=Path(args.output_dir, "topics.txt"),
            n=args.topic_words_to_save,
        )
        if args.save_all_topics:
            for i, topic_terms_i in enumerate(lda.training_topics):
                save_topics(
                    topic_terms_i,
                    fpath=Path(args.output_dir, "topics", f"{i}.txt"),
                    n=args.topic_words_to_save,
                )
    else:
        est_beta = lda.get_topics()
        topic_terms = [word_probs.argsort()[::-1] for word_probs in est_beta]
        npmi = npmi_scorer.compute_npmi(topics=topic_terms, n=args.eval_words)
        save_topics(
            [[inv_vocab[w] for w in topic[:100]] for topic in topic_terms],
            fpath=Path(args.output_dir, "topics.txt"),
            n=args.topic_words_to_save,
        )
        np.save(Path(args.output_dir, "beta.npy"), est_beta)

    tu = compute_tu(topic_terms, n=args.eval_words)
    to, overlaps = compute_to(topic_terms, n=args.eval_words, return_overlaps=True)
    n_overlaps = int(np.sum(overlaps == args.eval_words))
    metrics = {
        #'npmi': npmi,
        'npmi_mean': np.mean(npmi),
        #'tu': tu,
        'tu_mean': np.mean(tu),
        'to': to,
        'entire_overlaps': n_overlaps,
    }
    save_json(metrics, Path(args.output_dir, "metrics.json"))

    return lda, metrics

if __name__ == "__main__":

    parser = configargparse.ArgParser(
        description="parse args",
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    # Data
    parser.add("-c", "--config", is_config_file=True, default=None)
    parser.add("--input_dir", default=None)
    parser.add("--temp_output_dir", default=None, help="Temporary model storage during run, when I/O bound")

    parser.add("--output_dir", required=True, default=None)
    parser.add("--train_path", default="train.dtm.npz")
    parser.add("--eval_path", default="val.dtm.npz")
    parser.add("--vocab_path", default="vocab.json")
    parser.add("--eval_split", default=None, type=float)

    # Model-specific hyperparams
    parser.add("--num_topics", default=None, type=int)
    parser.add("--model", default="mallet", choices=["mallet", "gensim"])
    
    parser.add("--alpha", default=None)
    parser.add("--iterations", default=None, type=int)

    ## Gensim-only
    parser.add("--eta", default=None)
    parser.add("--passes", default=1, type=int)
    parser.add("--decay", default=0.5, type=float)

    ## Mallet-only
    parser.add("--beta", default=0.01, type=float)
    parser.add("--optimize_interval", type=int, default=0)
    parser.add("--mallet_path", default=None)
    
    # Evaluation
    parser.add("--eval_words", default=10, type=int)
    parser.add("--optimize_for_coherence", action="store_true", default=False)
    parser.add("--topic_words_to_save", default=50, type=int)
    parser.add("--save_all_topics", action="store_true", default=True)

    # Run settings
    parser.add("--run_seeds", default=[42], type=int, nargs="+", help="Seeds to use for each run")
    parser.add("--workers", default=4, type=int)
    args = parser.parse_args()
    if args.optimize_for_coherence:
        raise DeprecationWarning("Optimizing for coherence no longer supported.")
    if args.model == "gensim":
        args.alpha = "symmetric" if args.alpha is None else args.alpha
        args.eta = "symmetric" if args.eta is None else args.eta
        args.iterations = 50 if args.iterations is None else args.iterations

        try:
            args.alpha = float(args.alpha)
        except ValueError:
            args.alpha = args.alpha

        try:
            args.eta = float(args.eta)
        except ValueError:
            args.eta = args.eta

        
    if args.model == "mallet":
        args.alpha = 5.0 if args.alpha is None else args.alpha
        args.iterations = 1000 if args.iterations is None else args.iterations

    # Run for each seed
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(exist_ok=True, parents=True)

    for i, seed in enumerate(args.run_seeds):
        # make subdirectories for each run
        if len(args.run_seeds) == 1:
            output_dir = base_output_dir
        else:
            output_dir = Path(base_output_dir, str(seed))
            output_dir.mkdir(exist_ok=True, parents=True)

        args.seed = seed
        args.output_dir = output_dir
        if args.temp_output_dir: # if using, say, scratch space: reassign output dir
            args.output_dir = Path(args.temp_output_dir, str(np.random.randint(1000)))
            args.output_dir.mkdir(exist_ok=True, parents=True)
    
        # train
        print(f"\nOn run {i} of {len(args.run_seeds)}")
        model, metrics = main(args)

        # remove temp files to save space
        Path(args.output_dir, "corpus.txt").unlink(missing_ok=True)
        Path(args.output_dir, "corpus.mallet").unlink(missing_ok=True)
        
        if args.temp_output_dir: # copy from scratch back to the original
            shutil.copytree(args.output_dir, output_dir, dirs_exist_ok=True)
            shutil.rmtree(args.output_dir) # remove temporary directory

    # Aggregate results
    if len(args.run_seeds) > 1:
        agg_run_results = []
        for seed in args.run_seeds:
            output_dir = Path(base_output_dir, str(seed))
            try:
                metrics = load_json(Path(output_dir, "metrics.json"))
            except FileNotFoundError:
                continue
            metrics.pop("npmi")
            metrics.pop("tu")
            agg_run_results.append(metrics)

        agg_run_results_df = pd.DataFrame.from_records(agg_run_results)
        agg_run_results_df.to_csv(Path(base_output_dir, "run_results.csv"))
        print(
            f"\n=== Results over {len(args.run_seeds)} runs ===\n"
            f"Mean NPMI: "
            f"{agg_run_results_df.npmi_mean.mean():0.4f} ({agg_run_results_df.npmi_mean.std():0.4f}) "
        )
