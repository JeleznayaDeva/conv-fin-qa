import json
import math
from pathlib import Path
from typing import Optional

from conv_fin_qa.ask_llm import ask_financial_question
from conv_fin_qa.preprocessing import DataHandler
from conv_fin_qa.settings import PREPROCESSED_DATA_PATH


def run_evaluation(tables_only: bool = True, num_samples: Optional[int] = None) -> None:
    """Runs evalution on the model's performance with either tables only or full context.

    Parameters
    ----------
    tables_only
        flag for tables only context passed to the model
    num_samples
        number of samples to evaluate
    """
    data = json.load(Path(PREPROCESSED_DATA_PATH).open())
    all_documents = data.keys()
    comparison_margin = 0.5
    matches = 0
    running_idx = 0
    for document in all_documents:
        data_handler = DataHandler(document=data[document])
        context = data_handler.format_context(tables_only=tables_only)
        question = data_handler.default_question()
        expected_answer = data_handler.default_answer()
        response = ask_financial_question(context=context, question=question)
        try:
            if math.isclose(float(response), float(expected_answer), abs_tol=comparison_margin):
                matches += 1
        except ValueError:
            print(f"Expected answer: {expected_answer}")
        running_idx += 1
        if running_idx % 40 == 0:
            try:
                running_accuracy = matches / running_idx
            except ZeroDivisionError:
                running_accuracy = 0
            print(f"Running accuracy: {matches}/{running_idx} ({running_accuracy:.2f}) for tables_only={tables_only}")
        if num_samples and running_idx >= num_samples:
            break
    print(f"Performance: {matches}/{running_idx} ({running_accuracy:.2f}) for tables_only={tables_only}")
    return
