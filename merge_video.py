import os
from moviepy.editor import VideoFileClip, clips_array, ColorClip

# Specify the output directory
output_dir = 'C:/Users/sumuk/Downloads/camvid/CamVid/output_video_combined/'

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Load the two video files (ensure the correct file extension)
video1 = VideoFileClip('C:/Users/sumuk/Downloads/camvid/CamVid/output_video.mp4')
video2 = VideoFileClip('C:/Users/sumuk/Downloads/camvid/CamVid/output_video_pred.mp4')

# Resize both videos to the same height (optional)
video1 = video1.resize(height=500)
video2 = video2.resize(height=500)

# Combine the videos side by side
final_video = clips_array([[video1, video2]])

# Set the target duration for the video (10 seconds)
target_duration = 10

# If the video is shorter than the target duration, add black frames
if final_video.duration < target_duration:
    # Create a black screen of the same size as the video
    black_clip = ColorClip(size=final_video.size, color=(0, 0, 0), duration=target_duration - final_video.duration)

    # Append the black clip at the end
    final_video = clips_array([[video1, video2]]).fx('concat', black_clip)

# Set custom FPS (e.g., 30 FPS)
fps = 30

# Specify the output file path
output_video_path = os.path.join(output_dir, 'combined_video.mp4')

# Write the final video to the specified directory with custom fps
final_video.write_videofile(output_video_path, codec='libx264', fps=fps)

print(f"Video saved to: {output_video_path}")
