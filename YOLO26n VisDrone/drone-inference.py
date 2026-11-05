"""Object detection with optional tracking and line counting (Ultralytics YOLO26).

Examples:
    python drone-inference.py --source drone-video-2.mp4 --weights yolo26n-drone.pt
    python drone-inference.py --source drone-video-2.mp4 --track --weights yolo26n-drone.pt
    python drone-inference.py --source drone-video-2.mp4 --count --weights yolo26n-drone.pt
"""

import cv2
import argparse
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import colors


class DroneVision:
    """Detect, optionally track, and line-count objects with a YOLO model."""

    def __init__(self, args):
        """Load the model and reset the counting state."""
        self.args = args
        self.model = YOLO(args.weights)
        self.names = self.model.names

        self.line = None                 # ((x1, y1), (x2, y2)) in pixels
        self.scale = None                # (thickness, font, font_thickness)
        self.prev = {}                   # track id -> previous centre
        self.counts = defaultdict(int)   # class id -> crossings
        self.in_count = 0
        self.out_count = 0

    def run(self):
        """Stream the source, annotate every frame, and show or save it."""
        a = self.args
        if a.count:
            a.track = True

        stream = dict(source=a.source, conf=a.conf, iou=a.iou, imgsz=a.imgsz,
                      device=a.device, stream=True, verbose=False)
        results = (self.model.track(persist=True, tracker=a.tracker, **stream)
                   if a.track else self.model.predict(**stream))

        writer = None
        title = "Ultralytics YOLO26 " + ("tracking" if a.track else "detection")

        for res in results:
            frame = res.orig_img
            h, w = frame.shape[:2]

            # sizes and counting line depend on resolution, so set them once
            if self.scale is None:
                short = min(w, h)
                self.scale = (max(1, round(short / 360)),
                              max(0.4, short / 1200),
                              max(1, round(short / 600)))
                if a.line:
                    x1, y1, x2, y2 = (float(v) for v in a.line.split(","))
                else:
                    x1, y1, x2, y2 = 0.5, 0.0, 0.5, 1.0  # vertical centre line
                self.line = ((int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)))

            self._annotate(frame, res.boxes)

            if a.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(a.out, fourcc, 30, (w, h))
                writer.write(frame)

            if not a.no_show:
                cv2.imshow(title, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or Esc
                    break

        if writer is not None:
            writer.release()
            print(f"[i] saved to {a.out}")
        cv2.destroyAllWindows()

    def _annotate(self, frame, boxes):
        """Draw the line and boxes, tallying any crossings along the way."""
        a = self.args
        line, font, ft = self.scale
        p1, p2 = self.line

        def side(u, v, p):
            return (v[0] - u[0]) * (p[1] - u[1]) - (v[1] - u[1]) * (p[0] - u[0])

        if a.count:
            cv2.line(frame, p1, p2, ((108, 27, 255)), line + 1)
            for p in (p1, p2):
                cv2.circle(frame, p, line + 3, (108, 27, 255), -1)

        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.int().cpu().tolist()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(cls)
            conf = None if a.track else boxes.conf.cpu().tolist()

            for i, c in enumerate(cls):
                x1, y1, x2, y2 = xyxy[i].astype(int)
                tid = ids[i]
                color = colors(c, True)

                # while tracking we show the id and class; confidence is a
                # detection-only signal, so it shows up there instead
                label = f"#{tid} {self.names[c]}" if a.track else f"{self.names[c]} {conf[i]:.2f}"

                if a.count and tid is not None:
                    cur = ((x1 + x2) / 2, (y1 + y2) / 2)
                    prev = self.prev.get(tid)
                    self.prev[tid] = cur
                    if (prev is not None
                            and (side(p1, p2, prev) > 0) != (side(p1, p2, cur) > 0)
                            and (side(prev, cur, p1) > 0) != (side(prev, cur, p2) > 0)):
                        self.counts[c] += 1
                        if side(p1, p2, cur) >= 0:
                            self.in_count += 1
                        else:
                            self.out_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line, cv2.LINE_AA)
                (tw, th), base = cv2.getTextSize(label, 0, font, ft)
                pad = max(3, round(th * 0.35))
                cv2.rectangle(frame, (x1, max(0, y1 - th - base - pad)),
                              (x1 + tw + 2 * pad, y1), color, -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1 + pad, y1 - base), 0,
                            font, (255, 255, 255), ft, cv2.LINE_AA)

        if a.count:
            self._panel(frame)

    def _panel(self, frame):
        """Draw the running per-class and in/out counts in a corner panel."""
        _, font, ft = self.scale
        font *= 1.0
        margin, pad, gap, col_gap = (round(k * font) for k in (18, 14, 15, 24))

        rows = [("LINE COUNTING", "", (0, 255, 255))]
        rows += [(self.names[c], str(n), (255, 255, 255)) for c, n in sorted(self.counts.items())]
        rows.append(("IN", str(self.in_count), (120, 255, 120)))
        rows.append(("OUT", str(self.out_count), (120, 200, 255)))

        meas = {s: cv2.getTextSize(s, 0, font, ft)[0] for r in rows for s in r[:2] if s}
        ascent = max(meas[r[0]][1] for r in rows)
        row_h = ascent + gap
        left_w = max(meas[r[0]][0] for r in rows)
        right_w = max((meas[r[1]][0] for r in rows if r[1]), default=0)

        x0 = y0 = margin
        x1 = x0 + 2 * pad + left_w + col_gap + right_w
        y1 = y0 + 2 * pad + len(rows) * row_h - gap

        roi = frame[y0:y1, x0:x1]
        roi[:] = (roi * 0.5).astype(np.uint8)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)

        y = y0 + pad
        for left, right, color in rows:
            baseline = y + ascent
            cv2.putText(frame, left, (x0 + pad, baseline), 0, font, color, ft, cv2.LINE_AA)
            if right:
                cv2.putText(frame, right, (x1 - pad - meas[right][0], baseline), 0,
                            font, color, ft, cv2.LINE_AA)
            y += row_h


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Ultralytics YOLO inference")
    p.add_argument("--weights", default="yolo26s-drone.pt", help="path to trained model")
    p.add_argument("--source", required=True, help="image, video, folder or webcam index")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference size")
    p.add_argument("--device", default=None, help="cuda device, e.g. 0 or cpu")

    p.add_argument("--track", action="store_true", help="enable object tracking")
    p.add_argument("--count", action="store_true", help="enable line counting")
    p.add_argument("--tracker", default="bytetrack.yaml", help="tracker config")
    p.add_argument("--line", default=None,
                   help="counting line x1,y1,x2,y2 in 0-1 fractions "
                        "(default: vertical centre line)")

    p.add_argument("--save", action="store_false", help="write annotated video to disk")
    p.add_argument("--no-show", action="store_true", help="don't open a display window")
    p.add_argument("--out", default="output.mp4", help="output path when --save")
    return p.parse_args()


if __name__ == "__main__":
    DroneVision(parse_args()).run()
