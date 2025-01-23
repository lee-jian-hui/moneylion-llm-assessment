import logging


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
