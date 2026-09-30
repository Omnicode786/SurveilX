"""Bounded read-only HTTP ranges for ZIP member acquisition without full archive storage."""

import io
import re
import urllib.request


class RemoteZip(io.RawIOBase):
    def __init__(self, url, budget=1024**3):
        super().__init__()
        self.url, self.budget, self.transferred, self.position = url, budget, 0, 0
        self.size, self.etag = None, None
        self.cache_start, self.cache = 0, b""
        self._range(0, 0)

    def _range(self, first, last):
        expected = last - first + 1
        if expected > 32 * 1024**2 or self.transferred + expected > self.budget:
            raise ValueError("Remote ZIP transfer budget exceeded")
        headers = {"Range": f"bytes={first}-{last}", "User-Agent": "SurveilX-bounded-acquisition/1.0"}
        if self.etag:
            headers["If-Range"] = self.etag
        with urllib.request.urlopen(
            urllib.request.Request(self.url, headers=headers), timeout=60
        ) as response:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (first, last):
                raise ValueError("Server did not honor the exact bounded byte range")
            etag, size = response.headers.get("ETag"), int(match[3])
            if not etag or etag.startswith("W/"):
                raise ValueError("A stable strong archive ETag is required")
            if self.etag and (etag != self.etag or size != self.size):
                raise ValueError("Archive changed during acquisition")
            self.size, self.etag = size, etag
            content = response.read(expected + 1)
            if len(content) != expected:
                raise ValueError("Truncated or oversized range response")
            self.transferred += len(content)
            return content

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = (
            offset
            if whence == io.SEEK_SET
            else self.position + offset
            if whence == io.SEEK_CUR
            else self.size + offset
        )
        if position < 0 or position > self.size:
            raise ValueError("Seek outside archive")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if not size:
            return b""
        if self.cache_start <= self.position and self.position + size <= self.cache_start + len(self.cache):
            offset = self.position - self.cache_start
            content = self.cache[offset : offset + size]
        else:
            self.cache_start = self.position
            amount = min(max(size, 65536), self.size - self.position)
            self.cache = self._range(self.position, self.position + amount - 1)
            content = self.cache[:size]
        self.position += len(content)
        return content
