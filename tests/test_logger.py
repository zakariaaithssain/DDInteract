"""Tests for logging configuration."""

import sys

sys.path.insert(0, "src")

import logging
from pathlib import Path


def test_logger_name():
    from logger import logger

    assert logger.name == "ddi"


def test_logger_level():
    from logger import logger

    assert logger.level == logging.DEBUG


def test_logger_has_stream_handler():
    from logger import logger

    handlers = logger.handlers
    stream_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) >= 1


def test_logger_has_file_handler():
    from logger import logger

    handlers = logger.handlers
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) >= 1


def test_logger_stream_handler_level():
    from logger import logger

    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            assert h.level == logging.INFO
            return
    assert False, "No StreamHandler found"


def test_logger_file_handler_level():
    from logger import logger

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            assert h.level == logging.DEBUG
            return
    assert False, "No FileHandler found"


def test_logs_directory_created():
    assert Path("logs").is_dir()


def test_logger_exports_name():
    from logger import __all__

    assert "logger" in __all__
