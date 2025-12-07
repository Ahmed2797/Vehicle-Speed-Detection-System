from ultralytics import YOLO
import supervision as sv
import numpy as np
import argparse
import cv2
import time
from collections import defaultdict, deque

SOURCE = np.array([
                [-180,719],
                [1700,719],
                [805,288],
                [396,277],
                ])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransform:
    def __init__(self,source:np.ndarray,target:np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.M = cv2.getPerspectiveTransform(src=source,dst=target)

    def transform_points(self,points:np.ndarray)->np.ndarray:
        reshaped_points = points.reshape(-1,1,2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(src=reshaped_points,m=self.M)
        if transformed_points is None:
            print("❌ perspectiveTransform returned None")
            return points

        return transformed_points.reshape(-1,2)


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Speed Detection Estimator with YOLO and Supervision (Professional)'
    )
    parser.add_argument("--video", type=str, required=True, help="Path of the input video file")

    return parser.parse_args()


if __name__=='__main__':
    arg = arg_parser()
    model = YOLO('yolo11n.pt')
    video_info = sv.VideoInfo.from_video_path(arg.video)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
    thinkness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    bounding_box_annotation = sv.BoxCornerAnnotator(thickness=2)
    label_annotation = sv.LabelAnnotator(text_scale=text_scale,
                                         text_thickness=thinkness,
                                         text_position=sv.Position.BOTTOM_CENTER,
                                         text_color=sv.Color.GREY,
                                         smart_position=True)
    trace_annotation = sv.TraceAnnotator(thickness=thinkness,
                                         trace_length=video_info.fps*2,
                                         position=sv.Position.BOTTOM_CENTER,
                                         color_lookup=sv.ColorLookup.TRACK)
    
    SOURCE = np.array([[-180,719],
                       [1700,719],
                       [805,288],
                       [396,277]])  # example polygon
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 250
    TARGET = np.array([[0, 0],
                       [TARGET_WIDTH - 1, 0],
                       [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
                       [0, TARGET_HEIGHT - 1],])
    poly_zone = sv.PolygonZone(polygon=SOURCE)
    view_transform = ViewTransform(source=SOURCE,target=TARGET)
    cordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    frame_gen = sv.get_video_frames_generator(arg.video)
    for frame in frame_gen:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[poly_zone.trigger(detections=detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        points = view_transform.transform_points(points=points)

        labels=[]
        for track_id,[_,y] in zip(detections.tracker_id,points):
            cordinates[track_id].append(y)
            if len(cordinates[track_id]) < video_info.fps/2:
                labels.append(f'#{track_id}')
            else:
                cordinate_start = cordinates[track_id][-1]
                cordinate_end = cordinates[track_id][0]
                distance = abs(cordinate_start - cordinate_end)
                time = len(cordinates[track_id]) / video_info.fps
                speed = (distance / time) * 3.6
                labels.append(f"{int(speed)} km/h")
 

        # labels = [f"X:{round(x)} Y:{round(y)}" for [x,y] in points]
        # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        # labels = [
        # f"#{tracker_id}-{class_name}:{confidence:.2f}"
        # for tracker_id, class_name, confidence
        # in zip(detections.tracker_id, detections['class_name'], detections.confidence)
        # ]

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(scene=annotated_frame,
                                          polygon=SOURCE,
                                          color=sv.Color.RED,
                                          thickness=thinkness)
        annotated_frame = trace_annotation.annotate(scene=annotated_frame,
                                                    detections=detections)
        annotated_frame = bounding_box_annotation.annotate(scene = annotated_frame,
                                                           detections=detections)
        annotated_frame = label_annotation.annotate(scene=annotated_frame,
                                                    detections=detections,
                                                    labels=labels)
        
        cv2.imshow('Speed Annotation',annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()





## python main.py --video speed/data/vehicles1280x720.mp4