from ultralytics import YOLO
import supervision as sv
import numpy as np
import argparse
import cv2
import time


from speed.utilits import * 

if __name__ =="__main__":

    arg = arg_parser()
    video_info = sv.VideoInfo.from_video_path(arg.video)
    print(video_info.resolution_wh)
    fps = video_info.fps if video_info.fps and video_info.fps > 0 else 25.0

    model = YOLO("yolo11n.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps,track_activation_threshold=0.30)

    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
    
    roundbox_annote =sv.BoxAnnotator(thickness=1)
    label_anotator = sv.LabelAnnotator(text_scale=text_scale,text_thickness=thickness,text_position=sv.Position.BOTTOM_CENTER)

    #SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

    polyzone_pts = np.array([[0,583],
                             [1280,510], # 512
                             [805,288],
                             [396,277],
                             #[805,288]
                             ])  # example polygon
    
    poly_zone = sv.PolygonZone(polyzone_pts)

    h = build_homography()
    prev_positions = {}   # track_id -> (x_m, y_m, timestamp)
    frame_idx = 0


    frame_generator = sv.get_video_frames_generator(arg.video)

    for frame in frame_generator:
        frame_idx += 1
        t_now = time.time()


        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[poly_zone.trigger(detections)] 
        detections = byte_track.update_with_detections(detections)
        print(detections)


        # Labels for annotator
        labels = []
        for i in range(len(detections.tracker_id)):
            tid = detections.tracker_id[i]
            x1, y1, x2, y2 = detections.xyxy[i]
            # CORRECT bottom-center calculation:
            bottom_center = ((x1 + x2) / 2.0, float(y2))

            # Transform bottom-center to top-down target
            tx, ty = transform_point(h, bottom_center)  # in target pixels

            # Convert target pixels to meters (vertical scaling)
            meters_per_px = 1.0 # default 1.0 (1 px in target == 1 meter)
            x_m = tx * meters_per_px
            y_m = ty * meters_per_px

            # compute speed if previous exists
            speed_kph = None
            if tid in prev_positions:
                prev_xm, prev_ym, prev_t = prev_positions[tid]
                dt_sec = t_now - prev_t if (t_now - prev_t) > 0 else (1.0/fps)
                speed_mps = compute_speed_mps((prev_xm, prev_ym), (x_m, y_m), dt_sec)
                speed_kph = speed_mps * 3.6

                # draw speed text near box
                cv2.putText(frame, f"ID:{tid} {speed_kph:.1f} km/h",
                            (int(x1), max(int(y1)-12,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # Optionally annotate with ID only
                cv2.putText(frame, f"ID:{tid}", (int(x1), max(int(y1)-12,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # update prev_positions
            prev_positions[tid] = (x_m, y_m, t_now)

            # For BEV visualization: draw a small circle on bev_canvas
            bx = int(round(tx))
            by = int(round(ty))
            # if 0 <= by < bev_canvas_shape[0] and 0 <= bx < bev_canvas_shape[1]:
            #     cv2.circle(bev_canvas, (bx, by), 3, (0,255,255), -1)

            # prepare label list
            labels.append(f"#{tid}")

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame,polyzone_pts)
        anotated_frame = roundbox_annote.annotate(scene=annotated_frame,
                                                detections=detections)
        anotated_frame = label_anotator.annotate(scene=anotated_frame,
                                                detections=detections,
                                                labels=labels)
        
        cv2.imshow('Annotated_Frame:',anotated_frame)
        
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


## python ptc.py --video speed/data/vehicles1280x720.mp4


    
    