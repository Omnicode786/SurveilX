"""Bounded, shared JPEG encoding for snapshot and continuous camera previews."""
import threading

import cv2


class PreviewEncoder:
    def __init__(self):
        self.lock = threading.Lock()
        self.key = None
        self.jpeg = None

    def encode(self, capture):
        with self.lock:
            with capture.lock:
                if capture.frame is None:
                    return None, capture.sequence
                key = (capture.sequence, capture.last_service)
                if key == self.key:
                    return self.jpeg, key[0]
                frame = capture.frame.copy()
                outputs = list(capture.outputs)
            h, w = frame.shape[:2]
            for result in outputs:
                x1, y1, x2, y2 = result["box"]
                cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (130, 230, 200), 2)
                cv2.putText(frame, f"{result['label']} #{result['track_id']}",
                            (int(x1*w), max(15, int(y1*h)-8)), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (130, 230, 200), 1)
            success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                return None, key[0]
            self.key, self.jpeg = key, jpeg.tobytes()
            return self.jpeg, key[0]
