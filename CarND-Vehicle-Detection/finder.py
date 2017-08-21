import cv2
import numpy as np
from featurizer import extract_features


class CarFinder:
    def __init__(self, clf_pickle):
        self.colorspace = clf_pickle["colorspace"]
        self.orient = clf_pickle["orient"]
        self.pix_per_cell = clf_pickle["pix_per_cell"]
        self.cell_per_block = clf_pickle["cell_per_block"]
        self.hog_channel = clf_pickle["hog_channel"]
        self.clf = clf_pickle["clf"]
        self.scaler = clf_pickle["scaler"]

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, clf, X_scaler, orient, pix_per_cell, cell_per_block, ystart, ystop, scale):

        bbox_list = []

        # draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        # ch1 = img_tosearch[:,:,0]
        # ch2 = img_tosearch[:,:,1]
        # ch3 = img_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (img_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (img_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1
        # nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute HOG features for the entire image
        features = extract_features([img_tosearch], cspace=self.colorspace, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=self.hog_channel)[0]

        # hog1 = extract_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                sub_features = [f[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                                for f in features]
                sub_features = np.hstack(sub_features)
                # hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                # hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                # hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                # subimg = cv2.resize(img_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                # spatial_features = bin_spatial(subimg, size=spatial_size)
                # hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                # test_features = X_scaler.transform(
                #    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_features = X_scaler.transform(sub_features.reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    bbox = ((xbox_left, ytop_draw + ystart),
                            (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                    bbox_list.append(bbox)
                    # cv2.rectangle(draw_img, bbox[0],
                    #               bbox[1], (0, 0, 255), 6)

        return bbox_list

    def draw_bboxes(self, img, bbox_list):
        draw_img = np.copy(img)
        for bbox in bbox_list:
            cv2.rectangle(draw_img, bbox[0],
                          bbox[1], (0, 0, 255), 6)
        return draw_img

    def find(self, img):
        bbox_list = []
        # find positive bounding boxes
        bbox_list += self.find_cars(img, self.clf, self.scaler, self.orient, self.pix_per_cell, self.cell_per_block,
                                    ystart=400, ystop=500,
                                    scale=1.25)
        bbox_list += self.find_cars(img, self.clf, self.scaler, self.orient, self.pix_per_cell, self.cell_per_block,
                                    ystart=400, ystop=656,
                                    scale=1.5)
        #bbox_list += self.find_cars(img, self.clf, self.scaler, self.orient, self.pix_per_cell, self.cell_per_block,
        #                            ystart=450, ystop=700,
        #                            scale=2.0)
        return bbox_list
