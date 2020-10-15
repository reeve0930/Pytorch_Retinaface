import cv2
import numpy as np
import requests
from reelib import vision


class RetinaClient:
    def __init__(self, host):
        self.host = host

    def detect(self, frame):
        img = vision.encode_img(frame)
        data = {"image": img}
        results = requests.post(self.host, json=data)

        return results

    def write_bbox(self, frame, results, confidence=True, landmark=True, vis_thresh=0.6):
        frame_copy = np.copy(frame)
        for r in results:
            if float(r["confidence"]) < vis_thresh:
                continue
            text = "{:.4f}".format(float(r["confidence"]))
            cv2.rectangle(frame_copy, (r["point"]["x1"], r["point"]["y1"]),
                          (r["point"]["x2"], r["point"]["y2"]), (0, 0, 255), 2)
            if confidence:
                cx = r["point"]["x1"]
                cy = r["point"]["y1"] + 12
                cv2.putText(frame_copy, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            if landmark:
                cv2.circle(
                    frame_copy, (r["landmark"]["p1_x"], r["landmark"]["p1_y"]), 1, (0, 0, 255), 4)
                cv2.circle(
                    frame_copy, (r["landmark"]["p2_x"], r["landmark"]["p2_y"]), 1, (0, 0, 255), 4)
                cv2.circle(
                    frame_copy, (r["landmark"]["p3_x"], r["landmark"]["p3_y"]), 1, (0, 0, 255), 4)
                cv2.circle(
                    frame_copy, (r["landmark"]["p4_x"], r["landmark"]["p4_y"]), 1, (0, 0, 255), 4)
                cv2.circle(
                    frame_copy, (r["landmark"]["p5_x"], r["landmark"]["p5_y"]), 1, (0, 0, 255), 4)
        return frame_copy

if __name__=="__main__":
    client = RetinaClient("http://localhost:30000")

    frame = cv2.imread("curve/test.jpg")
    results = client.detect(frame)

    frame = client.write_bbox(frame, results.json())
    cv2.imwrite("test.jpg", frame)
