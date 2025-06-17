from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL
import os

def setup_logger(name, level=DEBUG, log_to_file=True):
    logger = getLogger(name)
    logger.setLevel(level)
    
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Stream handler for console output
    stream_handler = StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    
    if not logger.hasHandlers():
        logger.addHandler(stream_handler)
        
        if log_to_file:
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            file_handler = FileHandler(os.path.join(logs_dir, f'{name}.log'))
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger