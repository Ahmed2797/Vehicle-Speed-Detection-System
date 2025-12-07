

# speed
import numpy as np
import cv2

# ধরো, ভিডিও fps
fps = 30

# ধরো known reference: লেনের প্রস্থ 3.5 m এবং ভিডিওতে pixel প্রস্থ = 100 px
scale = 3.5 / 100  # meter per pixel

# tracker_id অনুযায়ী centroid রাখার জন্য dictionary
previous_centroids = {}

# ভিডিও লোড
cap = cv2.VideoCapture("vehicles1280x720.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # উদাহরণ হিসেবে detect করা গাড়ির bounding box (x1, y1, x2, y2)
    # সাধারণত YOLO detections থেকে পাওয়া যায়
    detections = [
        {"tracker_id": 1, "bbox": [100, 300, 200, 400]},
        {"tracker_id": 2, "bbox": [500, 320, 600, 420]}
    ]

    for det in detections:
        tid = det["tracker_id"]
        x1, y1, x2, y2 = det["bbox"]
        # centroid
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if tid in previous_centroids:
            prev_cx, prev_cy = previous_centroids[tid]
            dx = cx - prev_cx
            dy = cy - prev_cy
            distance_pixels = np.sqrt(dx**2 + dy**2)
            
            # pixel → meter
            distance_m = distance_pixels * scale
            
            # time between frames
            delta_time = 1 / fps
            
            # speed in m/s
            speed_m_s = distance_m / delta_time
            
            # convert to km/h
            speed_kmh = speed_m_s * 3.6
            
            # দেখানো
            cv2.putText(frame, f"ID {tid}: {speed_kmh:.1f} km/h", (int(cx), int(cy)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        # update previous centroid
        previous_centroids[tid] = (cx, cy)

    cv2.imshow("Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()





"""
speed_homography.py

ব্যবহার:
python speed_homography.py --video input.mp4 --out output.mp4

এই স্ক্রিপ্টটি:
- একটি হোমোগ্রাফি H হিসাব করে (SRC -> DST)
- প্রতিটি detection bottom-center পয়েন্টকে টার্গেটে রূপান্তর করে
- টার্গেট y কোঅর্ডিনেটকে meters ধরে (এই উদাহরণে 1 px = 1 m)
- দুই কনসেকিউটিভ ফ্রেমে একই object (id ধরে) অবস্থান পরিবর্তন থেকে speed হিসাব করে
"""

import cv2
import numpy as np
import argparse
import time

def arg_parser():
    p = argparse.ArgumentParser(description="Homography + speed estimation example")
    p.add_argument("--video", type=str, required=True, help="Input video file")
    p.add_argument("--out", type=str, default="out.mp4", help="Output video (with overlay)")
    return p.parse_args()

def build_homography():
    # SOURCE points (pixels) - change if you have more accurate values
    src = np.float32([
        [1252, 787],   # A
        [2298, 803],   # B
        [5039, 2159],  # C
        [-550, 2159]   # D
    ])

    # TARGET points (pixels) - design so 1 px ~= 1 m in both axes
    dst = np.float32([
        [0,   0],     # A'
        [24,  0],     # B'  (width 25 px -> 0..24)
        [24, 249],    # C'  (height 250 px -> 0..249)
        [0,  249]     # D'
    ])

    H = cv2.getPerspectiveTransform(src, dst)
    return H

def transform_point(H, pt):
    """ pt = (x,y) in source image pixels.
        returns (x', y') in target pixel coordinates (normalized) """
    pts = np.array([[pt]], dtype=np.float32)   # shape (1,1,2)
    dst = cv2.perspectiveTransform(pts, H)     # shape (1,1,2)
    x, y = float(dst[0,0,0]), float(dst[0,0,1])
    return x, y

def pixel_to_meter(y_px):
    """In our target design 1 px == 1 meter in vertical direction."""
    # If you set target height 250 px to represent 250 m, then:
    return y_px  # meters

def compute_speed_mps(p1_m, p2_m, dt):
    """p1_m and p2_m are (x_m, y_m) in meters in target coords. dt in seconds."""
    dx = p2_m[0] - p1_m[0]
    dy = p2_m[1] - p1_m[1]
    dist = np.sqrt(dx*dx + dy*dy)
    if dt <= 0:
        return 0.0
    return dist / dt

def main():
    args = arg_parser()
    H = build_homography()
    print("Homography H:\n", H)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer to save annotated vid
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # For demonstration, we assume you have detections per frame.
    # Replace this with your YOLO detections loop. Each detection must give:
    # bbox = (x1,y1,x2,y2), class_id, track_id  (if you use tracker)
    # For simplicity here we will simulate or you should plug in real detections.

    prev_positions = {}   # dict track_id -> (x_m, y_m, timestamp)

    frame_idx = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_now = time.time()
        dt = 1.0 / fps  # approximate per-frame time; or use (t_now - prev_time)
        prev_time = t_now

        # --- TODO: REPLACE this block with your detection+tracking results ---
        # Example fake detection list: list of dicts with keys bbox and id
        # In real code, get detections from your YOLO+tracker pipeline
        detections = []  # e.g. [{'id':1, 'bbox':(x1,y1,x2,y2)}, ...]
        # --------------------------------------------------------------------

        # For each detection: compute bottom-center, transform, compute speed
        for det in detections:
            tid = det['id']
            x1,y1,x2,y2 = det['bbox']
            bottom_center = ( (x1+x2)/2.0, y2 )  # pixel in source image

            tx, ty = transform_point(H, bottom_center)   # target pixels
            x_m = tx   # because 1 px = 1 m in our target
            y_m = pixel_to_meter(ty)

            # compute speed if previous exists
            if tid in prev_positions:
                prev_xm, prev_ym, prev_t = prev_positions[tid]
                dt_secs = dt  # or t_now - prev_t
                speed_mps = compute_speed_mps((prev_xm, prev_ym), (x_m, y_m), dt_secs)
                speed_kph = speed_mps * 3.6

                # Draw speed on frame
                cv2.putText(frame, f"ID:{tid} {speed_kph:.1f} km/h",
                            (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                speed_kph = None

            # update previous
            prev_positions[tid] = (x_m, y_m, t_now)

            # draw detection and mapped point (visual aid)
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
            # project the transformed location back onto the original image for display:
            # We can draw the source bottom center and optionally the corresponding topview coords.
            cv2.circle(frame, (int(bottom_center[0]), int(bottom_center[1])), 4, (0,255,255), -1)

        writer.write(frame)
        frame_idx += 1

        # If you want imshow (you said you need), uncomment:
        # cv2.imshow("Annotated", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    writer.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
