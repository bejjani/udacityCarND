import glob
import pickle
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
import itertools
from finder import CarFinder
from heat import Heat


def main(mode='test', video=None):
    # load classifier
    clfs = []

    #clfs.append(pickle.load(open("clf_pickle_YCrCb_ALL.p", "rb")))
    #clfs.append(pickle.load(open("clf_pickle_YCrCb_1.p", "rb")))
    #clfs.append(pickle.load(open("clf_pickle_YCrCb_2.p", "rb")))
    #clfs.append(pickle.load(open("clf_pickle_YUV_1.p", "rb")))
    clfs.append(pickle.load(open("clf_pickle_YUV_2.p", "rb")))

    clfs.append(pickle.load(open("clf_pickle_YCrCb_2_linear.p", "rb")))
    clfs.append(pickle.load(open("clf_pickle_YUV_1_linear.p", "rb")))
    #clfs.append(pickle.load(open("clf_pickle_YUV_2_linear.p", "rb")))

    heat_threshold = 12

    finders = [CarFinder(clf) for clf in clfs]
    heat = Heat()

    # helper function for video streams
    def pipeline(img):
        # find boxes
        bbox_list = list(
            itertools.chain(
                *[finder.find(img) for finder in finders])
        )
        # apply heatmap
        labels, _ = heat.apply(img, bbox_list, heat_threshold)
        return heat.draw_labeled_bboxes(np.copy(img), labels)

    if mode == 'test':
        # test image
        for fname in glob.glob('test_images/*.jpg'):
            print(fname)
            img = mpimg.imread(fname)

            # find boxes
            bbox_list = list(
                itertools.chain(
                    *[finder.find(img) for finder in finders])
            )
            out_img = finders[0].draw_bboxes(img, bbox_list)
            write_name = 'output_images/boxes/' + fname
            mpimg.imsave(write_name, out_img)

            # apply heatmap
            labels, heatmap = heat.apply(img, bbox_list, heat_threshold)
            draw_img = heat.draw_labeled_bboxes(np.copy(img), labels)
            write_name = 'output_images/heatmap/' + fname
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            plt.savefig(write_name)
            plt.close()

    elif mode == 'video':
        write_name = 'output_videos/' + video
        clip = VideoFileClip(video)
        video_clip = clip.fl_image(pipeline)
        video_clip.write_videofile(write_name, audio=False)


if __name__ == "__main__":
    if sys.argv[1] == 'video':
        # main('video', 'project_video.mp4')
        assert len(sys.argv) == 3, 'missing video argument'
        main(sys.argv[1], sys.argv[2])
    elif sys.argv[1] == 'test':
        main(sys.argv[1])