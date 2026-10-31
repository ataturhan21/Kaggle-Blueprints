import contextlib
import io
import unittest

from logger.logger import setup_applevel_logger


class LoggerTests(unittest.TestCase):
    def test_timestamp_does_not_include_literal_slice_syntax(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            logger = setup_applevel_logger(logger_name="blueprint.timestamp.test")
            logger.info("ready")
        self.assertRegex(
            output.getvalue(),
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - blueprint\.timestamp\.test - INFO - ready\n$",
        )


if __name__ == "__main__":
    unittest.main()
