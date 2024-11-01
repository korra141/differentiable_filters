import cv2
import os
import argparse
import pdb
import re 

def generate_video_from_images(image_folder, output_video_folder, image_prefix, fps=30):
    # Get list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.startswith(image_prefix) and img.endswith(".png") or img.endswith(".jpg")]
    images.sort(key = lambda x: int(re.findall(r'\d+',x.split('_')[-1])[0]))
    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'XVID '
    output_video = os.path.join(output_video_folder, f"{image_prefix}.mp4")
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video}")
    return output_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a video from images.')
    parser.add_argument('--image_folder', type=str,
                        default='/home/mila/r/ria.arora/scratch/HarmonicExponentialBayesFitler/diff_filter/output/w378s1vx/fig',
                        help='Path to the folder containing images.')
    parser.add_argument('--image_prefix_list', nargs='+', default=['s1_hef_0_traj_0'],
                        help='List of prefixes of the image files.')
    parser.add_argument('--output_video_folder', type=str,
                        help='Name of the output video file.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the video.')

    args = parser.parse_args()

    for image_prefix in args.image_prefix_list:
        # pdb.set_trace()
        if not args.output_video_folder.endswith(".mp4"):
            output_video = os.path.join(args.output_video_folder, f"{image_prefix}.mp4")
        else:
            output_video = args.output_video_folder
        generate_video_from_images(args.image_folder, output_video, image_prefix, args.fps)
