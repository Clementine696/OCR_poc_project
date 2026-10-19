"""Two copies of the app at once: each on its own port, one run log between them.

On Windows a second copy used to bind the FIRST one's port (Werkzeug sets
SO_REUSEADDR, which there allows it) and requests went to either at random. And
the run log's lock was a thread lock, so two processes rewriting `runs.csv` lost
each other's rows -- 40 of 600 appends and 58 of 60 deletes in the run that
found it.
"""
import csv
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import config

ROOT = Path(__file__).resolve().parent.parent


def _listener():
    """A socket listening the way Werkzeug's does: SO_REUSEADDR set."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    return sock


class PortTests(unittest.TestCase):
    def test_a_listening_port_is_not_free(self):
        with _listener() as sock:
            port = sock.getsockname()[1]
            self.assertFalse(app.port_free("127.0.0.1", port))
        self.assertTrue(app.port_free("127.0.0.1", port))

    def test_unset_port_steps_to_the_next_free_one(self):
        with _listener() as sock:
            taken = sock.getsockname()[1]
            with patch.object(config, "PORT", taken), \
                 patch.object(config, "PORT_FIXED", False), \
                 patch.object(config, "HOST", "127.0.0.1"), \
                 patch.object(app, "port_free", side_effect=lambda h, p: p != taken), \
                 patch.object(app, "say"):
                self.assertEqual(app.choose_port(), taken + 1)

    def test_a_set_port_that_is_taken_exits(self):
        with _listener() as sock:
            taken = sock.getsockname()[1]
            with patch.object(config, "PORT", taken), \
                 patch.object(config, "PORT_FIXED", True), \
                 patch.object(config, "HOST", "127.0.0.1"), \
                 patch.object(app, "say"):
                with self.assertRaises(SystemExit):
                    app.choose_port()


HAMMER = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, os.getcwd())
    import runlog
    role, n = sys.argv[1], int(sys.argv[2])
    for i in range(n):
        name = f"{role}-{os.getpid()}-{i}.pdf"
        assert runlog.record({"page_count": 1}, {"name": name})
        if role == "delete":
            mine = [r for r in runlog._read_all() if r["file"] == name]
            out = runlog.delete([{"_row": r["_row"], "timestamp": r["timestamp"],
                                  "file": r["file"]} for r in mine])
            assert out and out["deleted"] == 1, out
""")


class SharedLogTests(unittest.TestCase):
    def test_two_processes_lose_no_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "hammer.py"
            script.write_text(HAMMER, encoding="utf-8")
            env = {**os.environ, "OCR_LOG_DIR": str(Path(tmp) / "logs")}
            jobs = [subprocess.Popen([sys.executable, str(script), role, str(n)],
                                     cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                    for role, n in (("append", 60), ("append", 60), ("delete", 20))]
            for job in jobs:
                _, err = job.communicate(timeout=120)
                self.assertEqual(job.returncode, 0, err.decode(errors="replace"))
            with open(Path(tmp) / "logs" / "runs.csv", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 120)
        self.assertFalse([r for r in rows if r["file"].startswith("delete-")])


if __name__ == "__main__":
    unittest.main()
