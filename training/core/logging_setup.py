"""
Logging configuration and filters.
"""
import logging
import os

class _SuppressCapacityLogs(logging.Filter):
    """Filter to suppress MoE capacity logging messages."""
    def filter(self, record):
        msg = record.getMessage()
        return 'Capacity =' not in msg and 'real-time capacity-factor' not in msg

def setup_logging():
    """Set up logging configuration."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    # Add capacity filter to suppress MoE logs
    logging.getLogger().addFilter(_SuppressCapacityLogs())
    
    return logging.getLogger(__name__)

def setup_basic_logging():
    """Set up basic logging for main process."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
