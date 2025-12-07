import numpy as np
import argparse
import cv2 




def arg_parser():
    parser = argparse.ArgumentParser(
        description='Speed Detection Estimator with YOLO and Supervision (Professional)'
    )
    parser.add_argument("--video", type=str, required=True, help="Path of the input video file")
    # parser.add_argument("--out", type=str, default="out.mp4", help="Path to save processed output video")
    # parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model weights")
    # parser.add_argument("--show", action="store_true", help="Show windows (imshow)")
    # parser.add_argument("--bev", action="store_true", help="Show BEV (bird-eye-view) window")
    # parser.add_argument("--meters_per_px", type=float, default=1.0, help="Scaling: meters per target pixel (vertical)")
    # parser.add_argument(
    #     "--confidence_threshold",
    #     default=0.3,help="Confidence threshold for the model",type=float,)
    # parser.add_argument(
    #     "--iou_threshold",
    #     default=0.7, help="IOU threshold for the model", type=float)

    return parser.parse_args()


def build_homography(src_pts=None, dst_pts=None):
    """
    src_pts: list/array of 4 source points (image pixels) in order [A,B,C,D]
    dst_pts: list/array of 4 target points (top-down pixels) in order [A',B',C',D']
    returns H (3x3)
    """
    if src_pts is None:
        # default example 
        src = np.float32([
            [1252, 787],   # A (top-left of road region)
            [2298, 803],   # B (top-right of road region)
            [5039, 2159],  # C (bottom-right)
            [-550, 2159]   # D (bottom-left)
        ])
    else:
        src = np.float32(src_pts)

    if dst_pts is None:
        # target: choose width ~ 25 px for top 25m, height 250 px for 250m -> 1 px ~= 1 m
        dst = np.float32([
            [0,   0],     # A'
            [24,  0],     # B'
            [24, 249],    # C'
            [0,  249]     # D'
        ])
    else:
        dst = np.float32(dst_pts)

    H = cv2.getPerspectiveTransform(src, dst)
    return H  # src, dst


def transform_point(H, pt):
    """Transform single (x,y) using homography H. Returns (x', y')."""
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
