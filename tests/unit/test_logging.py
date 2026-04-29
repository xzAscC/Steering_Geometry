"""Tests for configure_logging utility function."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import pytest

from steering_geometry.utils import configure_logging

LOGGER_NAME = "steering_geometry"


@pytest.fixture(autouse=True)
def _cleanup_logger() -> Generator[None, None, None]:
    """Remove all handlers from the named logger after each test."""
    yield
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


class TestConfigureLoggingCreatesLogFile:
    """configure_logging should create a log file on disk."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        """Calling configure_logging with log_dir and log_name creates the file."""
        configure_logging(level="DEBUG", log_dir=tmp_path, log_name="test.log")

        log_file = tmp_path / "test.log"
        assert log_file.exists()
        assert log_file.stat().st_size >= 0


class TestConfigureLoggingWritesToFile:
    """configure_logging should route log messages to the file handler."""

    def test_writes_to_file(self, tmp_path: Path) -> None:
        """After configure_logging, logger.info() should appear in the log file."""
        configure_logging(level="DEBUG", log_dir=tmp_path, log_name="test.log")
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("hello from test")

        log_file = tmp_path / "test.log"
        content = log_file.read_text()
        assert "hello from test" in content


class TestConfigureLoggingConsoleOutput:
    """configure_logging should attach a StreamHandler with the correct level."""

    def test_console_output(self) -> None:
        """StreamHandler should exist on the logger after configure_logging."""
        configure_logging(level="DEBUG")

        logger = logging.getLogger(LOGGER_NAME)
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1
        assert stream_handlers[0].level == logging.DEBUG


class TestConfigureLoggingIdempotent:
    """configure_logging should not add duplicate handlers on repeated calls."""

    def test_idempotent(self) -> None:
        """Calling configure_logging twice should not duplicate handlers."""
        configure_logging(level="DEBUG")
        configure_logging(level="DEBUG")

        logger = logging.getLogger(LOGGER_NAME)
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert len(stream_handlers) == 1


class TestConfigureLoggingLevelFiltering:
    """configure_logging should respect the specified log level."""

    def test_level_filtering(self, tmp_path: Path) -> None:
        """With level=WARNING, info messages should be suppressed in file."""
        configure_logging(level="WARNING", log_dir=tmp_path, log_name="test.log")
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("should_not_appear")
        logger.warning("should_appear")

        log_file = tmp_path / "test.log"
        content = log_file.read_text()
        assert "should_not_appear" not in content
        assert "should_appear" in content


class TestConfigureLoggingDefaultLevel:
    """configure_logging should default to INFO when level is not specified."""

    def test_default_level(self) -> None:
        """Calling configure_logging without level should use INFO."""
        configure_logging()

        logger = logging.getLogger(LOGGER_NAME)
        assert logger.level == logging.INFO


class TestConfigureLoggingCreatesLogDir:
    """configure_logging should auto-create the log_dir if it doesn't exist."""

    def test_creates_log_dir(self, tmp_path: Path) -> None:
        """Passing a non-existent directory should create it automatically."""
        nested_dir = tmp_path / "deeply" / "nested" / "logs"
        assert not nested_dir.exists()

        configure_logging(level="DEBUG", log_dir=nested_dir, log_name="test.log")

        assert nested_dir.exists()
        assert (nested_dir / "test.log").exists()
