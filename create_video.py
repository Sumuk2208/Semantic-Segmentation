import cv2
import os

def images_to_video(input_dir, output_file, fps):
    images = sorted([img for img in os.listdir(input_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print("No images found in the directory.")
        return

    first_image_path = os.path.join(input_dir, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print("Error reading the first image.")
        return
    height, width, layers = frame.shape

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Use XVID codec and higher quality for the video
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    if not video.isOpened():
        print("Error: Unable to open video writer.")
        return

    for image in images:
        image_path = os.path.join(input_dir, image)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        # Resize image if necessary to ensure consistent resolution
        frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    print(f"Video saved as {output_file}")

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
input_directory = r"C:\Users\sumuk\PycharmProjects\CSCI635_project\saved_masks"
output_video = "C:/Users/sumuk/Downloads/camvid/CamVid/output_video/output_video_pred.mp4"
frames_per_second = 30

images_to_video(input_directory, output_video, frames_per_second)

# View the video
play_video(output_video)
