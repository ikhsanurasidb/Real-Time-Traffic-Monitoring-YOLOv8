import argparse
from typing import List
import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config 

import supervision as sv

model = YOLO("best-yolov8.pt")

class CustomSink:
    def __init__(self, zone_configuration_path: str, classes: List[int]):
        self.classes = classes
        self.class_names = ['bus', 'car', 'motorcycle', 'truck']
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.8)
        self.fps_monitor = sv.FPSMonitor()
        self.line = load_zones_config(file_path=zone_configuration_path) 
        self.line_zone = sv.LineZone(
                                        start=sv.Point(self.line[0][0][0], self.line[0][0][1]), 
                                        end=sv.Point(self.line[0][1][0], self.line[0][1][1]),
                                        triggering_anchors=(sv.Position.CENTER,)
                                    )

        self.line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
        self.box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
        self.bus_in_count = 0
        self.bus_out_count = 0
        self.car_in_count = 0
        self.car_out_count = 0
        self.motor_in_count = 0
        self.motor_out_count = 0
        self.truck_in_count = 0
        self.truck_out_count = 0

    def on_prediction(self, detections: sv.Detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        valid_detections_mask = find_in_list(detections.class_id, self.classes)
        detections = detections[valid_detections_mask]
        detections = self.tracker.update_with_detections(detections)
        detected_class = detections.class_id
        
        in_count, out_count = self.line_zone.trigger(detections)
        detected_object_in = in_count
        detected_object_out = out_count
        
        counted_object_in = [i for i, x in enumerate(detected_object_in) if x == True]
        in_count_forEach_class = [self.class_names[detected_class[i]] for i in counted_object_in]
        
        counted_object_out = [i for i, x in enumerate(detected_object_out) if x == True]
        out_count_forEach_class = [self.class_names[detected_class[i]] for i in counted_object_out]
        
        if len(in_count_forEach_class) > 0:
            for i in in_count_forEach_class:
                if i == 'bus':
                    self.bus_in_count += 1
                elif i == 'car':
                    self.car_in_count += 1
                elif i == 'motorcycle':
                    self.motor_in_count += 1
                elif i == 'truck':
                    self.truck_in_count += 1
        
        if len(out_count_forEach_class) > 0:
            for i in out_count_forEach_class:
                if i == 'bus':
                    self.bus_out_count += 1
                elif i == 'car':
                    self.car_out_count += 1
                elif i == 'motorcycle':
                    self.motor_out_count += 1
                elif i == 'truck':
                    self.truck_out_count += 1

        annotated_frame = frame.image.copy()
        
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"FPS: {fps:.1f}",
            text_anchor=sv.Point(45, 20),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Cars in : {self.car_in_count}",
            text_anchor=sv.Point(60, 60),
            background_color=sv.Color.from_hex("#ff2919"),
            text_color=sv.Color.from_hex("#000000"),
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Motos in: {self.motor_in_count}",
            text_anchor=sv.Point(60, 100),
            background_color=sv.Color.from_hex("#c97771"),
            text_color=sv.Color.from_hex("#000000"),
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Buses in: {self.bus_in_count}",
            text_anchor=sv.Point(60, 140),
            background_color=sv.Color.from_hex("#ffb60a"),
            text_color=sv.Color.from_hex("#000000"),
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Trucks in: {self.truck_in_count}",
            text_anchor=sv.Point(60, 180),
            background_color=sv.Color.from_hex("#ff700a"),
            text_color=sv.Color.from_hex("#000000"),
        )
        
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Cars out : {self.car_out_count}",
            text_anchor=sv.Point(1180, 60),
            background_color=sv.Color.from_hex("#ff2919"),
            text_color=sv.Color.from_hex("#000000"),
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Motos out: {self.motor_out_count}",
            text_anchor=sv.Point(1180, 100),
            background_color=sv.Color.from_hex("#c97771"),
            text_color=sv.Color.from_hex("#000000"),
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Buses out: {self.bus_out_count}",
            text_anchor=sv.Point(1180, 140),
            background_color=sv.Color.from_hex("#ffb60a"),
            text_color=sv.Color.from_hex("#000000"),
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Trucks out: {self.truck_out_count}",
            text_anchor=sv.Point(1180, 180),
            background_color=sv.Color.from_hex("#ff700a"),
            text_color=sv.Color.from_hex("#000000"),
        )
        
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:.2f}"
            for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        annotated_frame = self.line_zone_annotator.annotate(
            frame=annotated_frame,
            line_counter=self.line_zone
        )
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)

def main(
    rtsp_url: str,
    zone_configuration_path: str,
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    model = YOLO(weights)

    def inference_callback(frame: VideoFrame) -> sv.Detections:
        results = model(frame.image, verbose=False, conf=confidence, device=device)[0]
        return sv.Detections.from_ultralytics(results).with_nms(threshold=iou)

    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes)

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=rtsp_url,
        on_video_frame=inference_callback,
        on_prediction=sink.on_prediction,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using RTSP stream."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--rtsp_url",
        type=str,
        required=True,
        help="Complete RTSP URL for the video stream.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to the model weights file. Default is 'yolov8n.pt'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu', 'mps' or 'cuda'). Default is 'cpu'.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        rtsp_url=args.rtsp_url,
        zone_configuration_path=args.zone_configuration_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )
