import logging
import os
from datetime import datetime
from tqdm import tqdm
import atexit

class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Configure and return a logger that can be imported and used across modules.
    Integrates tqdm for console output
    """
    
    # Create logs directory if it doesn't exist
    if log_file is None:
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create logger
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(message)s')
    
    # Console handler using tqdm
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    
    # Error log file
    error_log_file = os.path.join(os.path.dirname(log_file), f'errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # Separate error logger
    error_logger = logging.getLogger('pipeline_errors')
    error_logger.setLevel(logging.ERROR)
    if error_logger.hasHandlers():
        error_logger.handlers.clear()
    error_logger.addHandler(error_handler)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log start timestamp directly to file
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Logging start: {start_time}")
    
    # Register function to log end timestamp at exit
    def log_end():
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\nLogging end: {end_time}")
    
    atexit.register(log_end)
    
    return logger, log_file, error_logger

# Create a default logger instance for direct imports
logger, log_file_path, error_logger = setup_logger()