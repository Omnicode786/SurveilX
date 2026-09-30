import io
import zipfile

import pytest

from scripts.remote_zip import RemoteZip
from scripts.prepare_sh17 import CLASSES, labels


def test_remote_zip_ranges_crc_and_cache(monkeypatch):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("labels/example.txt", "10 0.5 0.5 0.2 0.2")
    data = content.getvalue()

    class Response(io.BytesIO):
        status = 206

    def open_range(request, **kwargs):
        first, last = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        response = Response(data[first : last + 1])
        response.headers = {"Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": '"stable"'}
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_range)
    source = RemoteZip("https://example.test/archive", budget=10000)
    with zipfile.ZipFile(source) as archive:
        boxes, ids = labels(archive.read("labels/example.txt").decode())
    assert ids == [10] and CLASSES[10] == "helmet" and CLASSES[16] == "safety_vest"
    assert boxes[0] == pytest.approx([0.4, 0.4, 0.6, 0.6])


def test_non_range_response_is_rejected(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"whole archive"))
    with pytest.raises(ValueError, match="exact bounded"):
        RemoteZip("https://example.test/archive")
