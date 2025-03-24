from typing import List
import json


def preprocess_data(data_path: str):
    """
    Preprocess the data to a more convenient format:
    {
        "filename": {
            "pre_text": "Pre-table text",
            "post_text": "Post-table text",
            "table_ori": [
                ...
            ]
            ...
        },
        ...
    }
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    preprocessed_data = {}
    for item in data:
        preprocessed_data[item["filename"]] = item
    json.dump(preprocessed_data, open("data/train_preprocessed.json", "w"))


class DataHandler:
    def __init__(self, document: dict):
        self.document = document
        self.question_key = "qa"

    def pretty_table(self, table: List[List[str]]) -> str:
        """
        Pretty print the table by aligning the cells within columns.
        """
        pretty_table_str = ""
        columns = list(zip(*table))
        span_chars_per_column = [max(len(cell) for cell in column) for column in columns]

        for row in table:
            pretty_table_str += (
                "|" + "|".join(
                    cell.ljust(span_chars_per_column[cell_col_idx])
                    for cell_col_idx, cell in enumerate(row)
                ) + "|\n"
            )
        return pretty_table_str

    def format_context(self, tables_only: bool = True) -> str:
        """Get formatted context of the document."""
        table = self.document["table"]
        if not tables_only:
            pre_text_list = self.document["pre_text"] if not isinstance(self.document["pre_text"], str) else [self.document["pre_text"]]
            post_text_list = self.document["post_text"] if not isinstance(self.document["post_text"], str) else [self.document["post_text"]]
            pre_text = "\n".join(pre_text_list) + "\n"
            post_text = "\n".join(post_text_list) + "\n"
        else:
            pre_text = ""
            post_text = ""
        table_str = self.pretty_table(table)

        return f"{pre_text}{table_str}{post_text}"

    def default_question(self) -> str:
        """Used for evaluation mode only. Get the first question from the dataset"""
        try:
            question = self.document[self.question_key]["question"]
        except KeyError:
            try:
                question = self.document["qa_0"]["question"]
                self.question_key = "qa_0"
            except KeyError:
                raise ValueError("No question found in the document.")
        return question

    def default_answer(self) -> str:
        """used for evaluation mode only. Return the answer to the default question"""
        answer = self.document[self.question_key]["answer"]
        return answer.strip("%").replace(",", "")
