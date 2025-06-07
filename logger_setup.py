import time
import logging
import coloredlogs  # For colored console output
from typing import Dict, List, Any, Tuple, Union # Added Union

# Configure root logger first to catch any early messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

class RateLimitedLogger:
    def __init__(self, logger: logging.Logger, interval: float = 5.0):
        self.logger: logging.Logger = logger
        self.interval: float = interval
        self.last_log: Dict[str, float] = {}
        self.buffer: List[Union[Tuple[str, str], Tuple[str, str, str]]] = [] # Buffer can hold 2 or 3 item tuples
        self.buffer_size: int = 100
        self.last_flush: float = time.time()
        
    def info(self, message: str, key: Union[str, None] = None):
        current_time = time.time()
        if key is None:
            key = message
            
        if key not in self.last_log or (current_time - self.last_log[key]) >= self.interval:
            self.buffer.append(('info', message))
            self.last_log[key] = current_time
            
            # Flush buffer if it's full or enough time has passed
            if len(self.buffer) >= self.buffer_size or (current_time - self.last_flush) >= self.interval:
                self._flush_buffer()
            
    def error(self, message: str, key: Union[str, None] = None, **kwargs: Any):
        current_time = time.time()
        if key is None:
            key = message
            
        if key not in self.last_log or (current_time - self.last_log[key]) >= self.interval:
            self.buffer.append(('error', message))
            self.last_log[key] = current_time
            self._flush_buffer()  # Always flush errors immediately
            
    def warning(self, message: str, key: Union[str, None] = None):
        current_time = time.time()
        if key is None:
            key = message
            
        if key not in self.last_log or (current_time - self.last_log[key]) >= self.interval:
            self.buffer.append(('warning', message, key))
            self.last_log[key] = current_time
            
            # Flush buffer if it's full or enough time has passed
            if len(self.buffer) >= self.buffer_size or (current_time - self.last_flush) >= self.interval:
                self._flush_buffer()
    
    def debug(self, message: str, key: Union[str, None] = None):
        current_time = time.time()
        if key is None:
            key = message
            
        if key not in self.last_log or (current_time - self.last_log[key]) >= self.interval:
            self.buffer.append(('debug', message, key))
            self.last_log[key] = current_time
            
            # Flush buffer if it's full or enough time has passed
            if len(self.buffer) >= self.buffer_size or (current_time - self.last_flush) >= self.interval:
                self._flush_buffer()
    
    def setLevel(self, level: int) -> None:
        """Set the logging level of the underlying logger."""
        self.logger.setLevel(level)

    def _flush_buffer(self) -> None:
        if not self.buffer:
            return
            
        for item in self.buffer:
            if len(item) == 3:
                level, message, key = item
            else:
                # Handle older format for backward compatibility
                level, message = item
                
            if level == 'info':
                self.logger.info(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'debug':
                self.logger.debug(message)
                
        self.buffer.clear()
        self.last_flush = time.time()

# Initialize the logger when this module is imported
def setup_logger():
    try:
        # Silence external library logs
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

        # Get the 'main' logger
        logger = logging.getLogger('main')
        logger.handlers.clear()  # Remove any existing handlers
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Create console handler with color
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)

        # Create formatter and add it to the console handler
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
        console.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console)

        # Create file handler
        try:
            file_handler = logging.FileHandler('trading_run.log', mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter) # Use the same formatter for the file
            logger.addHandler(file_handler)
            logger.info("File logger initialized at trading_run.log")
        except Exception as e_file:
            logger.error(f"Failed to initialize file logger: {e_file}")
        
        # Initialize coloredlogs
        # Temporarily disable coloredlogs to isolate the stall issue
        # coloredlogs.install(
        #     level='DEBUG',
        #     logger=logger,
        #     fmt='%(asctime)s %(levelname)-8s %(message)s',
        #     datefmt='%H:%M:%S',
        #     level_styles={
        #         'debug':    {'color': 'cyan'},
        #         'info':     {'color': 'green'},
        #         'warning':  {'color': 'yellow'},
        #         'error':    {'color': 'red'},
        #         'critical': {'color': 'red', 'bold': True}
        #     },
        #     field_styles={
        #         'asctime':  {'color': 'blue'}
        #     }
        # )
        
        # Create rate-limited logger
        rate_limited_logger = RateLimitedLogger(logger)
        return rate_limited_logger
        
    except Exception as e:
        # Fallback to basic logging if there's an error
        logging.basicConfig(level=logging.INFO)
        _logger = logging.getLogger('main') # Use a different variable name to avoid confusion
        _logger.error(f"Error setting up logger: {str(e)}")
        return _logger

# Initialize the logger when this module is imported
logger: Union[RateLimitedLogger, logging.Logger] = setup_logger() # Global logger's type