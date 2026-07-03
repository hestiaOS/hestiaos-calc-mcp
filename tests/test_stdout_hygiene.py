"""Stdout hygiene test: MCP server must write ONLY JSON-RPC to stdout.

Start the server as a real subprocess, send MCP JSON-RPC requests,
capture stdout, and verify every line is valid JSON-RPC.
"""

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


class TestStdoutHygiene:
    """Real stdout/stderr capture of the MCP server subprocess."""

    def _start_server(self):
        """Start entrypoint.py as a subprocess and return handle."""
        proc = subprocess.Popen(
            [sys.executable, "entrypoint.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return proc

    def _send_request(self, proc, request: dict) -> tuple[bytes, bytes]:
        """Send a JSON-RPC request and read stdout/stderr with timeout."""
        line = json.dumps(request) + "\n"
        proc.stdin.write(line.encode())
        proc.stdin.flush()

        stdout_data = b""
        stderr_data = b""
        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:
            r, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.3)
            for fd in r:
                if fd == proc.stdout:
                    data = os.read(proc.stdout.fileno(), 65536)
                    if data:
                        stdout_data += data
                elif fd == proc.stderr:
                    data = os.read(proc.stderr.fileno(), 65536)
                    if data:
                        stderr_data += data
            # Stop when we have a complete JSON-RPC response (newline-terminated)
            if stdout_data and b"\n" in stdout_data:
                break

        return stdout_data, stderr_data

    def test_stdout_has_only_json_rpc(self):
        """Server stdout has exclusively JSON-RPC frames — no stray output."""
        proc = self._start_server()

        try:
            # 1. Send ping — verify clean JSON-RPC response
            stdout1, stderr1 = self._send_request(proc, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
            })
            lines1 = stdout1.split(b"\n")
            for line in lines1:
                line = line.strip()
                if not line:
                    continue
                # Each non-empty line must be valid JSON-RPC
                try:
                    data = json.loads(line)
                    assert "jsonrpc" in data, f"Non-JSON-RPC on stdout: {line}"
                    assert data["jsonrpc"] == "2.0"
                except json.JSONDecodeError:
                    pytest.fail(f"Non-JSON output on stdout: {line!r}")

            # 2. Send tools/call for calculate — triggers kernel worker spawn
            stdout2, stderr2 = self._send_request(proc, {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "calculate",
                    "arguments": {
                        "expression": "1/2+1/3",
                    },
                },
            })
            lines2 = stdout2.split(b"\n")
            for line in lines2:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    assert "jsonrpc" in data, f"Non-JSON-RPC on stdout: {line}"
                    assert data["jsonrpc"] == "2.0"
                except json.JSONDecodeError:
                    pytest.fail(f"Non-JSON output on stdout: {line!r}")

            # 3. Verify calculate returned correct result
            for line in lines2:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "result" in data and "content" in data.get("result", {}):
                    for item in data["result"]["content"]:
                        text = item.get("text", "")
                        if text:
                            result_data = json.loads(text)
                            if result_data.get("ok"):
                                assert result_data["result"]["rational"] == "5/6"
                                break

            # 4. Stderr has startup logs
            all_stderr = stderr1 + stderr2
            assert b"calc-mcp starting" in all_stderr or b"[calc-mcp]" in all_stderr, (
                f"Expected startup log on STDERR, got: {all_stderr[:300]}"
            )

        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
            from calc_mcp.core import _kill_worker
            _kill_worker()

    def test_first_calculate_after_init_returns_correct_result(self):
        """First kernel-relevant tools/call after subprocess start must
        produce a valid ok:true result — no Idris init output on stdout.

        Regression test for the kernel_worker.py warmup fix: without it,
        the first call triggers init_idris() which putStrLn's all_results
        onto stdout BEFORE the real result, causing the first response
        to contain a concatenated stale string like
        '5/6,1/6,1/3,3/4,8/27' instead of the actual calculation.
        """
        proc = self._start_server()

        try:
            # 1. MCP initialize handshake (does NOT trigger Idris kernel)
            stdout_init, _ = self._send_request(proc, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "regression-test", "version": "1.0"},
                },
            })
            init_data = json.loads(stdout_init.strip())
            assert init_data.get("id") == 1
            assert "result" in init_data, f"No result in init response: {stdout_init!r}"

            # 2. Send initialized notification (one-way, no response)
            proc.stdin.write(json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }).encode() + b"\n")
            proc.stdin.flush()

            # 3. FIRST kernel-relevant call: calculate("1+1")
            stdout1, _ = self._send_request(proc, {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "calculate",
                    "arguments": {"expression": "1+1", "precision": 10},
                },
            })

            lines1 = stdout1.split(b"\n")
            # Every non-empty line must be valid JSON-RPC
            result1 = None
            for line in lines1:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                assert data.get("jsonrpc") == "2.0", f"Non-JSON-RPC: {line!r}"
                if data.get("id") == 2:
                    content = data.get("result", {}).get("content", [])
                    if content:
                        payload = json.loads(content[0].get("text", "{}"))
                        result1 = payload

            assert result1 is not None, "No calculate result in first call"
            assert result1.get("ok") is True, (
                f"First calculate call returned ok=false: "
                f"{result1.get('error', 'no error')}. "
                f"Likely Idris init string on stdout."
            )
            assert result1["result"]["rational"] == "2", (
                f"First calculate call expected '2', got "
                f"'{result1['result'].get('rational')}'"
            )

            # 4. SECOND kernel call: calculate("1+1") again
            stdout2, _ = self._send_request(proc, {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "calculate",
                    "arguments": {"expression": "1+1", "precision": 10},
                },
            })

            lines2 = stdout2.split(b"\n")
            result2 = None
            for line in lines2:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                assert data.get("jsonrpc") == "2.0"
                if data.get("id") == 3:
                    content = data.get("result", {}).get("content", [])
                    if content:
                        payload = json.loads(content[0].get("text", "{}"))
                        result2 = payload

            assert result2 is not None, "No calculate result in second call"
            assert result2.get("ok") is True
            assert result2["result"]["rational"] == "2", (
                f"Second calculate call expected '2', got "
                f"'{result2['result'].get('rational')}'"
            )

        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
            from calc_mcp.core import _kill_worker
            _kill_worker()
