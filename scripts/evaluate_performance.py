from argparse import ArgumentParser

from conv_fin_qa.performance_evaluation import run_evaluation
from conv_fin_qa.preprocessing import preprocess_data
from conv_fin_qa.settings import DATA_PATH

parser = ArgumentParser()
parser.add_argument(
    "--preprocess_data",
    help="Whether to preprocess the data file",
    action="store_true",
)
args = parser.parse_args()

if __name__ == "__main__":
    # Preprocess data
    if args.preprocess_data:
        preprocess_data(data_path=DATA_PATH)

    # Evaluate the model's performance with context restricted to tables only
    run_evaluation(tables_structured=True, num_samples=100)

    # Evaluate the model's performance with full context
    run_evaluation(tables_structured=False, num_samples=100)
