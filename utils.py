import logging
import psutil
import torch
from langchain.prompts import PromptTemplate


def setup_logger(name: str, log_file: str = "main.log", level: int = logging.DEBUG):
    """
    Sets up a logger that logs to both the console and a log file.

    Parameters:
    - name (str): The name of the logger.
    - log_file (str): The file to log messages to (default: 'main.log').
    - level (int): The logging level (default: logging.DEBUG).

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()  # Logs to the console
    file_handler = logging.FileHandler(log_file)  # Logs to the file

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger



def log_resource_usage(logger: logging.Logger):
    logger.info(f"CPU memory usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")



def get_device(device_preference: str = "cpu"):
    """
    Determine if GPU (CUDA) is available and select the appropriate device.

    Args:
        device_preference (str): Preferred device ("cuda", "cpu", or "auto").
    Returns:
        str: The device to be used ("cuda:0" or "cpu").
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
        logger.info("Using GPU (CUDA) for model loading and inference.")
    else:
        device = "cpu"
        logger.info("Using CPU for model loading and inference.")
    return device


def truncate_conversation_history(
    conversation_history: str, 
    max_tokens_for_history: int
) -> str:
    """
    Truncate the oldest parts of the conversation history to fit within the maximum token limit.

    Args:
        conversation_history (str): The full conversation history.
        max_tokens_for_history (int): Maximum tokens allowed for the conversation history.

    Returns:
        str: Truncated conversation history.
    """
    history_lines = conversation_history.split("\n")
    truncated_history = []

    # Start from the end and add lines until we reach the token limit
    current_token_count = 0
    for line in reversed(history_lines):
        token_count = len(line.split())  # Approximation: count words as tokens
        if current_token_count + token_count > max_tokens_for_history:
            break
        truncated_history.append(line)
        current_token_count += token_count

    # Return the reversed truncated history (to maintain chronological order)
    return "\n".join(reversed(truncated_history))


def extend_prompt_template(base_template: PromptTemplate, additional_context: str) -> PromptTemplate:
    """
    Extend an existing PromptTemplate by appending additional context.

    Args:
        base_template (PromptTemplate): The original prompt template.
        additional_context (str): The additional instructions or context to append.

    Returns:
        PromptTemplate: A new prompt template with the added context.
    """
    # Combine the original template and additional context
    extended_template_string = base_template.template.strip() + "\n\n" + additional_context.strip()


    # Create a new PromptTemplate using the combined template string
    extended_template = PromptTemplate(
        input_variables=base_template.input_variables,
        template=extended_template_string
    )

    logger.info(f"modified template: {extended_template.template}")
    return extended_template


logger = setup_logger("utils", level=logging.INFO)
logger.info(f"get_device: {get_device()}")




import time

class BenchmarkReport:
    def __init__(self, context_window_size: int, prompt_template: str):
        self.context_window_size = context_window_size
        self.prompt_template = prompt_template
        self.start_time = time.time()
        self.questions = []
        self.answers = []
        self.errors = []

    def add_question_and_answer(self, question: str, answer: str):
        self.questions.append(question)
        self.answers.append(answer)

    def add_error(self, question: str, error: str):
        self.questions.append(question)
        self.errors.append(error)

    def generate_report(self) -> str:
        end_time = time.time()
        total_time = time.strftime("%H:%M:%S", time.gmtime(end_time - self.start_time))

        report_lines = [
            "Benchmark Report",
            "=" * 25,
            f"Context Window Size: {self.context_window_size}",
            f"Prompt Template: {self.prompt_template}",
            "",
        ]

        for i, question in enumerate(self.questions):
            report_lines.append(f"Question {i + 1}: {question}")
            if i < len(self.answers):
                report_lines.append(f"Answer {i + 1}: {self.answers[i]}")
            elif i < len(self.errors):
                report_lines.append(f"Error {i + 1}: {self.errors[i]}")
            report_lines.append("")

        report_lines.append(f"Time Taken: {total_time}")
        return "\n".join(report_lines)

    def save_to_file(self, output_file: str):
        with open(output_file, "w") as file:
            file.write(self.generate_report())
