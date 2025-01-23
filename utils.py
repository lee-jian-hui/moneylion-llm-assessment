import logging
import psutil
import torch



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



logger = setup_logger("utils", level=logging.INFO)
logger.info(f"get_device: {get_device()}")