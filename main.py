import cv2
from helmet_detector import HelmetDetector
from sound_manager import SoundManager

def main():
    # Initialize coqmponents
    detector = HelmetDetector(model_path="C:\\Users\\harsh\\PycharmProjects\\mahesh_helmet\\helmet-detection\\models\\yolov8m.pt\\yolov8m.pt")
    sound_manager = SoundManager()
    cap = cv2.VideoCapture(0)  # Laptop webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Helmet detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run helmet detection
        results = detector.detect(frame)
        annotated_frame = detector.draw_results(frame, results)

        # Check for helmet
        helmet_detected = False
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Class 0: helmet
                    helmet_detected = True
                    break
            if helmet_detected:
                break

        if not helmet_detected:
            print("No helmet detected! Beeping...")
            sound_manager.play_beep()

        # Display the frame
        cv2.imshow("Helmet Detection", annotated_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




# import cv2
# import av
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# from helmet_detector import HelmetDetector
# from sound_manager import SoundManager
#
#
# detector = HelmetDetector(model_path="C:\\Users\\harsh\\PycharmProjects\\mahesh_helmet\\helmet-detection\\models\\yolov8m.pt\\yolov8m.pt")
# sound_manager = SoundManager()
#
# st.title("Real-Time Helmet Detection")
#
# class HelmetVideoProcessor(VideoProcessorBase):
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         results = detector.detect(img)
#         annotated_frame = detector.draw_results(img, results)
#
#         helmet_detected = False
#         for result in results:
#             for box in result.boxes:
#                 if box.cls == 0:
#                     helmet_detected = True
#                     break
#             if helmet_detected:
#                 break
#
#         if not helmet_detected:
#             print("No helmet detected! Beeping...")
#             sound_manager.play_beep()
#
#         return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
#
# webrtc_streamer(
#     key="helmet-detection",
#     video_processor_factory=HelmetVideoProcessor,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,
# )