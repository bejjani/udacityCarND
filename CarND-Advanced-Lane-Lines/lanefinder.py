import numpy as np
import cv2
from line import Line
import matplotlib.pyplot as plt

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


class LaneFinder:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.left_line = Line()
        self.right_line = Line()

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the magnitude
        sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
        # Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        # Return this mask as your binary_output image
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def col_threshold(self, img, r_thresh=(0, 255), s_thresh=(0, 255)):
        # R channel
        r_channel = img[:, :, 2]
        r_binary = np.zeros_like(img[:, :, 0])
        r_binary[(r_channel > r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

        # S channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        # color_binary = np.dstack(( np.zeros_like(r_binary), r_binary, s_binary))

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(r_binary)
        combined_binary[(r_binary == 1) | (s_binary == 1)] = 1

        return combined_binary

    def undistort(self, img, mtx, dist):
        return cv2.undistort(img, mtx, dist, None, mtx)

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])

        # define 4 source points
        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        # define 4 destination points
        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        # use cv2.warpPerspective() to warp image to a top-down view
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        return warped, M, Minv

    def fit_line(self, img, left_fit=None, right_fit=None):
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if (left_fit is not None) & (right_fit is not None):
            # Assume you now have a new warped binary image
            # from the next frame of video (also called "img")
            # It's now much easier to find line pixels!
            margin = 100
            left_lane_inds = (
                (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
            right_lane_inds = (
                (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
            out_img = img
        else:
            # Create an output image to draw on and  visualize the result
            # Take a histogram of the bottom half of the image
            histogram = np.sum(img[np.int(img.shape[0] / 2):, :], axis=0)
            out_img = np.dstack((img, img, img)) #* 255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(img.shape[0] / nwindows)
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = img.shape[0] - (window + 1) * window_height
                win_y_high = img.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color=(0,1,0), thickness=2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color=(0,1,0), thickness=2)
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit, out_img

    def get_radius(self, ploty, leftx, rightx):
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5)\
                        / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5)\
                         / np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        return left_curverad, right_curverad

    def get_car_position(self, warped, left_fitx, right_fitx):
        cam_pos = (left_fitx[-1] + right_fitx[-1]) / 2
        return (cam_pos - warped.shape[1] / 2) \
               * xm_per_pix

    def draw(self, undist, warped, left_fitx, right_fitx, ploty, Minv, left_radius, right_radius, pos):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # add radius and position info
        cv2.putText(result, 'curve radius: {0:.2f}m'.format(left_radius),
                    org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(0, 255, 0))
        cv2.putText(result, 'vehicle offset: {0:.2f}m'.format(pos),
                    org=(50, 70),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.5,
                    color=(0, 255, 0))

        return result

    def find(self, img, verbose=False, fname=None):
        # Undistort the image
        undst = self.undistort(img, self.mtx, self.dist)
        # Thresholding
        mag_binary = self.mag_thresh(undst, mag_thresh=(30, 100))
        col_binary = self.col_threshold(undst, r_thresh=(220, 255), s_thresh=(150, 255))
        dir_binary = self.dir_threshold(undst, thresh=(0.7, 1.3))
        binary = np.zeros_like(dir_binary)
        binary[((mag_binary == 1) & (dir_binary == 1)) | (col_binary == 1)] = 1
        if verbose:
            write_name = 'output_images/thresholded/' + fname
            plt.imshow(binary, cmap="gray")
            plt.savefig(write_name)
            plt.close()
        # Perspective
        warped, M, Minv = self.unwarp(binary)
        if verbose:
            write_name = 'output_images/birds-eye/' + fname
            plt.imshow(warped, cmap="gray")
            plt.savefig(write_name)
            plt.close()
        # detect and fit poly to lines
        self.left_line.current_fit, self.right_line.current_fit, fitted_img = \
            self.fit_line(warped, self.left_line.current_fit, self.right_line.current_fit)
        ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
        left_fitx = self.left_line.current_fit[0] * ploty ** 2 + self.left_line.current_fit[1] * ploty + \
                    self.left_line.current_fit[2]
        right_fitx = self.right_line.current_fit[0] * ploty ** 2 + self.right_line.current_fit[1] * ploty + \
                     self.right_line.current_fit[2]
        if verbose:
            write_name = 'output_images/fitted/' + fname
            plt.imshow(fitted_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.savefig(write_name)
            plt.close()
        # Radius
        left_radius, right_radius = self.get_radius(ploty, left_fitx, right_fitx)
        # position
        pos = self.get_car_position(warped, left_fitx, right_fitx)
        # draw overlaid img
        return self.draw(img, warped, left_fitx, right_fitx, ploty, Minv, left_radius, right_radius, pos)
