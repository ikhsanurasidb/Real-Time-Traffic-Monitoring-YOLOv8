****Real-Time object detection application for traffic monitoring, especially in Indonesia.****

Detect vehicles including cars, motorbikes, buses, and trucks. In the absence of an IP camera, a custom RTSP server is utilized to simulate the transmission of real-time video. RTSP, a standard protocol, is used for streaming video data from IP cameras.

![SCR-20240603-hjwc](https://github.com/ikhsanurasidb/Real-Time-Traffic-Monitoring-YOLOv8/assets/151383202/d07820ee-d104-464e-979b-83a8987da161)


Libraries used:
- Ultralytics
- OpenCV
- roboflow/supervision
- roboflow/inference

Functionality to draw lines at desired locations within the application to facilitate the counting of objects that intersect with these lines.

<img width="636" alt="SCR-20240603-hgvz" src="https://github.com/ikhsanurasidb/Real-Time-Traffic-Monitoring-YOLOv8/assets/151383202/2efdb738-fcb3-454b-b5ab-f3c4ab2c4d0b">
