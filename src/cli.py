import argparse
import os, sys

from src.models.featurePolicy import DefaultFeatures
from src.models.engine import ModelEngine
from src.models.forest import RandomForest
from src.models.xgboost import XGBoost
from src.models.knn import KNN
from src.models.lstm import LSTM
from src.models.tree import DecisionTree
from src.models.logistic import LogisticRegression
from src.models.bilstm import BILSTM
from src.logger import Logger

logging = Logger()
logger = logging.get_logger()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    return False


def train_random_forest_model(*args, **kwargs):
    policy = DefaultFeatures()
    model = RandomForest()
    engine = ModelEngine(policy=policy, model=model)
    engine.build()
    
def train_xgboost_model(*args, **kwargs):
    policy = DefaultFeatures()
    model = XGBoost()
    engine = ModelEngine(policy=policy, model=model)
    engine.build()
    
def train_knn_model(*args, **kwargs):
    policy = DefaultFeatures()
    model = KNN()
    engine = ModelEngine(policy=policy, model=model)
    engine.build()

def train_lstm_model(*args, **kwargs):
    policy = DefaultFeatures()
    model = LSTM()
    engine = ModelEngine(policy=policy, model=model)
    engine.build()
    
def train_tree_model(*args, **kwargs):
    policy = DefaultFeatures()
    model = DecisionTree()
    engine = ModelEngine(policy=policy, model=model)
    engine.build()
    
def train_logistic_model(*args, **kwargs):
    policy = DefaultFeatures()
    model = LogisticRegression()
    engine = ModelEngine(policy=policy, model=model)
    engine.build()

def train_bilstm_model(*args, **kwargs):
    policy = DefaultFeatures()
    model = BILSTM()
    engine = ModelEngine(policy=policy, model=model)
    engine.build()

def make_shared_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="<LEVEL>",
        help="Set the logging level.\nChoices: [%(choices)s]\nDefault: %(default)s\n\n",
        default="INFO",
    )
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        dest="data_path",
        metavar="<PATH>",
        help="Path to ANP basin composite and conventional profile files.\nDefault: %(default)s\n\n",
        default="./downloads",
    )
    return parser


def make_train_subparsers() -> dict:
    train_subparsers = {}
    train_subparsers["random-forest"] = {
        "help": "Train a Random Forest model using the feature dataset",
        "description": "Create a model training engine to train a Random Forest model using the feature dataset.",
        "func": train_random_forest_model,
    }
    train_subparsers["xgboost"] = {
        "help": "Train an XGBoost model using the feature dataset",
        "description": "Create a model training engine to train an XGBoost model using the feature dataset.",
        "func": train_xgboost_model,
    }
    train_subparsers["knn"] = {
        "help": "Train a KNN model using the feature dataset",
        "description": "Create a model training engine to train a KNN model using the feature dataset.",
        "func": train_knn_model,
    }
    train_subparsers["lstm"] = {
        "help": "Train an LSTM model using the feature dataset",
        "description": "Create a model training engine to train an LSTM model using the feature dataset.",
        "func": train_lstm_model,
    }
    train_subparsers["decision-tree"] = {
        "help": "Train a Decision Tree model using the feature dataset",
        "description": "Create a model training engine to train a Decision Tree model using the feature dataset.",
        "func": train_tree_model,
    }
    train_subparsers["logistic-regression"] = {
        "help": "Train a Logistic Regression model using the feature dataset",
        "description": "Create a model training engine to train a Logistic Regression model using the feature dataset.",
        "func": train_logistic_model,
    }
    train_subparsers["bilstm"] = {
        "help": "Train a BiLSTM model using the feature dataset",
        "description": "Create a model training engine to train a BiLSTM model using the feature dataset.",
        "func": train_bilstm_model,
    }

    return train_subparsers

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLIDE: A command line tool for Smart Line Identification and Data Extraction"
    )
    parser = make_shared_commands(parser)

    subparsers = parser.add_subparsers(dest="command", required=True, help="Main commands")

    parser_train = subparsers.add_parser(
        "train",
        help="Train machine learning models using the feature dataset",
        description="Create a model training engine to train machine learning models using the feature dataset.",
    )
    
    train_subparsers = parser_train.add_subparsers(dest="train_type", required=True, help="Train subcommands")
    
    for name, opts in make_train_subparsers().items():
        sub = train_subparsers.add_parser(name, help=opts["help"], description=opts["description"])
        sub.set_defaults(func=opts["func"])

    return parser


def cli() -> None:
    """Main entry point for the slide command line interface."""
    parser = build_argparser()
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    args = parser.parse_args()
    args_dict = vars(args).copy()
    logging.set_level(args_dict.pop("log_level"))

    try:
        args.func(**args_dict)
    except Exception:
        raise
    os._exit(0)


if __name__ == "__main__":
    sys.argv = ["script.py", "-ll", "DEBUG", "scrap", "-s", "0"]
    cli()
