import argparse

import cv2
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(description="Render detections of a trained YOLO model into a video.")

    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the trained YOLO model (best.pt).")
    parser.add_argument("-v", "--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output video file.")

    return parser.parse_args()


def render_video(model: YOLO, video_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    # Same resolution and frame rate as the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect

        result = model(frame, verbose=False)
        annotated_frame = result[0].plot()
        cv2.imshow("Annotated Video", annotated_frame)
        writer.write(annotated_frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

def main():
    args = parse_arguments()

    model = YOLO(args.model) # TODO: Load your model

    render_video(
        model=model,
        video_path=args.video,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()