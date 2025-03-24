import json
from pathlib import Path
from conv_fin_qa.ask_llm import ask_financial_question
from conv_fin_qa.preprocessing import DataHandler
from conv_fin_qa.settings import PREPROCESSED_DATA_PATH


def main():
    """Main function to run the project as a module."""
    # Load the preprocessed data
    try:
        data = json.load(Path(PREPROCESSED_DATA_PATH).open())
    except FileNotFoundError:
        print("Please preprocess the data before evaluating performance")
        return

    # Ask the user for the document id and get the context
    document_id = input("Please enter the document id: ")

    data_handler = DataHandler(document=data[document_id])
    context = data_handler.format_context()

    # Print the context of the document and ask the user to enter a question
    print(f"Here is the context for the document {document_id}:\n")
    print(f"{context}")
    question = input("Please enter your question: ")
    if not question:
        question = input("Please enter a valid question: ")

    # Generate the response
    ask_financial_question(context=context, question=question)


if __name__ == "__main__":
    main()
