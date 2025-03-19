import logging
import os

def setup_logger(log_file="app.log", log_level=logging.INFO):
    """
    Set up a logger that logs messages to both the console and a file.

    Parameters:
        log_file (str): The name of the file where logs will be saved.
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logger (logging.Logger): Configured logger instance.
    """

    # Create a custom logger
    logger = logging.getLogger("AppLogger")
    logger.setLevel(log_level)

    # Prevent duplicate logs if the logger is already set up
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define the log format
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # File handler to log to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)

    # Console handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger initialized. Logging to file: %s", log_file)
    return logger

# if __name__ == "__main__":
    # Initialize the logger
    # logger = setup_logger(log_file="output.log", log_level=logging.INFO)

#     # Log some messages
#     logger.debug("This is a DEBUG message.")
#     logger.info("This is an INFO message.")
#     logger.warning("This is a WARNING message.")
#     logger.error("This is an ERROR message.")
#     logger.critical("This is a CRITICAL message.")
