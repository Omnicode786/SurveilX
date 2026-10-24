import threading
from types import SimpleNamespace
import cv2
import numpy as np
from surveilx.preview import PreviewEncoder


def test_preview_cache_invalidates_for_new_frame_or_new_detections(monkeypatch):
    capture=SimpleNamespace(lock=threading.Lock(), frame=np.zeros((64,96,3),np.uint8), sequence=1,
                            last_service=1, outputs=[])
    encoder=PreviewEncoder()
    original=cv2.imencode
    calls=[]
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(cv2,'imencode',counted)
    first, sequence=encoder.encode(capture)
    assert sequence==1 and first.startswith(b'\xff\xd8')
    assert encoder.encode(capture)[0] is first
    assert len(calls)==1
    capture.last_service=2
    capture.outputs=[{'box':[.1,.1,.8,.8],'label':'person','track_id':1}]
    assert encoder.encode(capture)[0]!=first
    capture.sequence=2
    encoder.encode(capture)
    assert len(calls)==3
    assert not capture.frame.any()  # Review overlay never mutates raw source/evidence.


def test_preview_waits_for_source_without_encoding():
    capture=SimpleNamespace(lock=threading.Lock(),frame=None,sequence=0)
    assert PreviewEncoder().encode(capture)==(None,0)


def test_continuous_preview_requires_authentication(client):
    assert client.get('/api/cameras/missing/live.mjpg').status_code==401


def test_stream_sends_unique_frames_and_stops_when_disconnected(monkeypatch):
    import asyncio
    from surveilx import api
    sequences=iter([1,1,2])
    capture=SimpleNamespace(stop_event=threading.Event(),preview=SimpleNamespace(
        encode=lambda _: (b'jpeg',next(sequences))))
    monkeypatch.setattr(api,'runtime',SimpleNamespace(cameras={'camera':capture}))
    class Request:
        calls=0
        async def is_disconnected(self):
            self.calls+=1
            return self.calls>3
    async def consume():
        response=await api.camera_live('camera',Request(),user=object())
        return [part async for part in response.body_iterator]
    parts=asyncio.run(consume())
    assert len(parts)==2
    assert b'X-Sequence: 1\r\n' in parts[0]
    assert b'X-Sequence: 2\r\n' in parts[1]
    assert all(b'Content-Length: 4\r\n' in part and part.endswith(b'jpeg\r\n') for part in parts)
