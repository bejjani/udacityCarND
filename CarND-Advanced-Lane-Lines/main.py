import pickle
import glob
import cv2
import sys
from moviepy.editor import VideoFileClip
from lanefinder import LaneFinder


def main(mode='test', video=None):
    # load calibration. Result of running `calibrate.py`
    dist_pickle = pickle.load(open('camera_cal/dist_pickle.p', 'rb'))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']

    # instantiate lane finder
    lane_finder = LaneFinder(mtx, dist)

    if mode == 'test':
        # iterate through test images
        images = glob.glob('test_images/*.jpg')
        for idx, fname in enumerate(images):
            print(fname)
            img = cv2.imread(fname)
            # detect lanes
            lane_finder.left_line.current_fit, lane_finder.right_line.current_fit = None, None
            dst = lane_finder.find(img, verbose=True, fname=fname)
            # overlay lanes over original and write to fs
            write_name = 'output_images/lane_detection/' + fname
            cv2.imwrite(write_name, dst)
    elif mode == 'video':
        # Name of output video after applying lane lines
        write_name = 'output_videos/' + video
        # Apply lane lines to each frame of video
        clip = VideoFileClip(video)
        video_clip = clip.fl_image(lane_finder.find)
        video_clip.write_videofile(write_name, audio=False)

if __name__ == "__main__":
    if sys.argv[1] == 'video':
        assert len(sys.argv) == 3, 'missing video argument'
        main(sys.argv[1], sys.argv[2])
    elif sys.argv[1] == 'test':
        main(sys.argv[1])
