import logging
from multiprocessing import Pipe


from warmpool._logging import PipeHandler, forward_subprocess_log
from warmpool import PoolWithTimeout

from ._helpers import log_message, log_with_exception


class TestPipeHandler:
    def test_sends_structured_dict(self):
        parent_connection, child_connection = Pipe()
        handler = PipeHandler(child_connection)
        logger = logging.getLogger("test.pipe")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.info("hello world")
            assert parent_connection.poll(timeout=1.0)
            tag, payload, meta = parent_connection.recv()
            assert tag == "log"
            assert meta == {}
            assert payload["level"] == "INFO"
            assert payload["message"] == "hello world"
            assert payload["logger"] == "test.pipe"
            assert "timestamp" in payload
            assert "process_id" in payload
        finally:
            logger.removeHandler(handler)
            parent_connection.close()
            child_connection.close()

    def test_exception_info(self):
        parent_connection, child_connection = Pipe()
        handler = PipeHandler(child_connection)
        logger = logging.getLogger("test.pipe.exc")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            try:
                raise RuntimeError("kaboom")
            except RuntimeError:
                logger.exception("caught it")
            tag, payload, _ = parent_connection.recv()
            assert "exception" in payload
            assert "kaboom" in payload["exception"]
        finally:
            logger.removeHandler(handler)
            parent_connection.close()
            child_connection.close()

    def test_levels(self):
        parent_connection, child_connection = Pipe()
        handler = PipeHandler(child_connection)
        logger = logging.getLogger("test.pipe.levels")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            for level_name in ("DEBUG", "INFO", "WARNING", "ERROR"):
                logger.log(getattr(logging, level_name), f"{level_name} msg")
            received = []
            while parent_connection.poll(timeout=0.5):
                _, payload, _ = parent_connection.recv()
                received.append(payload["level"])
            assert received == ["DEBUG", "INFO", "WARNING", "ERROR"]
        finally:
            logger.removeHandler(handler)
            parent_connection.close()
            child_connection.close()


class TestForwardSubprocessLog:
    def test_forward_basic(self, caplog):
        payload = {
            "timestamp": 1234567890.0,
            "level": "WARNING",
            "message": "something happened",
            "logger": "child.module",
            "process_id": 99999,
        }
        with caplog.at_level(logging.DEBUG, logger="warmpool.subprocess"):
            forward_subprocess_log(payload)
        assert "something happened" in caplog.text
        assert caplog.records[0].levelno == logging.WARNING

    def test_forward_preserves_extra(self, caplog):
        payload = {
            "timestamp": 1234567890.0,
            "level": "INFO",
            "message": "hi",
            "logger": "child.module",
            "process_id": 42,
        }
        with caplog.at_level(logging.DEBUG, logger="warmpool.subprocess"):
            forward_subprocess_log(payload)
        record = caplog.records[0]
        assert record.timestamp == 1234567890.0
        assert record.process_id == 42
        assert record.logger == "child.module"


class TestIntegrationLogs:
    def test_subprocess_logs_reach_parent(self, caplog):
        # Root logger must be DEBUG *before* the pool spawns so the worker
        # inherits the level and forwards DEBUG-level messages.
        root = logging.getLogger()
        orig_level = root.level
        root.setLevel(logging.DEBUG)
        try:
            pool = PoolWithTimeout(max_tasks=50, keep_spare=False)
            try:
                with caplog.at_level(logging.DEBUG, logger="warmpool.subprocess"):
                    pool.run(log_message, 5.0, message="hello from child")
                assert any("hello from child" in r.message for r in caplog.records)
            finally:
                pool.shutdown()
        finally:
            root.setLevel(orig_level)

    def test_subprocess_exception_log(self, pool, caplog):
        with caplog.at_level(logging.DEBUG, logger="warmpool.subprocess"):
            pool.run(log_with_exception, 5.0)
        assert any(
            "boom" in r.message
            or ("exception" in dir(r) and "boom" in getattr(r, "exception", ""))
            for r in caplog.records
        )
