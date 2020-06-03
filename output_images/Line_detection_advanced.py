import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def grad_thresh(img, thresh=(20,100)):
    # Gradient thresholds:
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

def colorHSV_thresh(img, thresh=(130,255)):
    # Color thresholds:
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary
def colorBGR_thresh(img, thresh=(200,255)):
    # Color thresholds:
    R_channel = img[:,:,2]
    # Threshold color channel
    R_binary = np.zeros_like(R_channel)
    R_binary[(R_channel >= thresh[0]) & (R_channel <= thresh[1])] = 1
    return R_binary

def warp(img, src, dst, img_size):

    #compute the perspective trasform, matrix M
    M = cv2.getPerspectiveTransform(src, dst)
    # Could compute the inverse by swapping the input parameters
    Minv = cv2.getPerspectiveTransform(dst, src)

    # creat warped image -uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size,flags=cv2.INTER_LINEAR)

    return warped, M, Minv

def find_lane_pixels(binary_warped):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # a=plt.figure()
    # plt.plot(histogram)
    # plt.ylabel('Histogram')
    # plt.xlabel('X (pixels)')
    # plt.savefig('histogram.png')#for readme
    # plt.show()

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 100#increased this number to avoid being misled by shades

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img, left_lane_inds, right_lane_inds, nonzeroy, nonzerox

def fit_polynomial(leftx, lefty, rightx, righty, size_binary_warped_0):

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    # to avoid estimating the part of line that there is no data, do not start from 0
    ind_0 = max(min(lefty),min(righty)) #size_binary_warped_0 - max(min(lefty),min(righty))
    #print(ind_0)
    ploty = np.linspace(ind_0, size_binary_warped_0-1, size_binary_warped_0 )

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return ploty, left_fitx, right_fitx, left_fit, right_fit

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def measure_off_center_real(left_fit_cr_0, right_fit_cr_0, img_size_0,xm_per_pix):
    lane_mid = (left_fit_cr_0 + right_fit_cr_0)/2
    middle_img = img_size_0/2
    car_off_center = (lane_mid - middle_img)*xm_per_pix

    return car_off_center

def unwarp(img, Minv, img_size):

    # creat warped image -uses linear interpolation
    unwarped = cv2.warpPerspective(img, Minv, img_size,flags=cv2.INTER_LINEAR)

    return unwarped

def draw_line(out_img, left_fitx, right_fitx, ploty):
    # only for 1d polynomial:
    # color=[255, 255, 0]
    # thickness=5
    # cv2.line(out_img, (left_fitx[0].astype(int), ploty[0].astype(int)),\
    #  (left_fitx[-1].astype(int), ploty[-1].astype(int)), color, thickness)
    # cv2.line(out_img, (right_fitx[0].astype(int), ploty[0].astype(int)),\
    #  (right_fitx[-1].astype(int), ploty[-1].astype(int)), color, thickness)
    # for curved polynomials:
    draw_points_left = (np.asarray([left_fitx, ploty]).T).astype(np.int32)   # needs to be int32 and transposed
    draw_points_right = (np.asarray([right_fitx, ploty]).T).astype(np.int32)   # needs to be int32 and transposed
    cv2.polylines(out_img, [draw_points_left], False, (0, 255,0),7)  # args: image, points, closed, color
    cv2.polylines(out_img, [draw_points_right], False, (0, 255,0),7)  # args: image, points, closed, color

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    return out_img

def search_around_poly(binary_warped, left_fit, right_fit,margin):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # # Fit new polynomials
    # left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    return leftx, lefty, rightx, righty, result #out_img

def visualize_detected_pixels(out_img, lefty, leftx, righty, rightx):
    """show detected pixels on the warped image"""
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img

def visualize_region_search_around_poly(binary_warped, left_lane_inds, right_lane_inds, left_fitx, right_fitx, margin_around_line, ploty, nonzeroy, nonzerox):
    """    draw region around poly on the road_box    """
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # plt.imshow(out_img)
    # plt.title('out_img', fontsize=10)
    # mpimg.imsave("out_img.png", out_img)
    # plt.show()

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin_around_line, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin_around_line,
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin_around_line, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin_around_line,
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # plt.imshow(result)
    # plt.title('result', fontsize=10)
    # plt.show()
    # mpimg.imsave("result.png", result)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return result

def visualize_window_serach(binary_warped_window_pixel, undist_road,Minv, img_size ):
    """     draw search windows on the road    """

    binary_warped_window_pixel_unwraped = unwarp(binary_warped_window_pixel, Minv, img_size)
    # plt.imshow(black_region_unwraped)
    # plt.title('black_region_unwraped', fontsize=10)
    # mpimg.imsave("black_region_unwraped.png", black_region_unwraped)
    # plt.show()

    road_window = cv2.addWeighted(undist_road, 1., binary_warped_window_pixel_unwraped, 0.8, 0.)
    # plt.imshow(road_region)
    # plt.show()
    # mpimg.imsave("road_region.png", road_region)#for readme

    return road_window

def visualize_lane(binary_warped,undist_road, ploty, left_fitx, right_fitx, Minv, img_size):
    """     Draw lane area on the road    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    #newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    newwarp = unwarp(color_warp, Minv, img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(undist_road, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    return result

def visualize_lines(undist_road, src, dst, img_size,left_fitx, right_fitx, ploty ):
    """    Draw lane lines on the road_line    """
    warped_road, M, Minv = warp(undist_road, src, dst, img_size)
    # plt.imshow(warped_road)
    # plt.title('warped road with rectangle', fontsize=10)
    # mpimg.imsave("warped_road.png", warped_road)#for readme
    # plt.show()

    # plot lines on the warped road image:
    black_wraped = np.zeros_like(warped_road)
    black_line_wraped = draw_line(black_wraped, left_fitx, right_fitx, ploty)
    # plt.imshow(black_line_wraped)
    # plt.title('black_line_wraped', fontsize=10)
    # mpimg.imsave("black_line_wraped.png", black_line_wraped)
    # plt.show()

    black_line_unwraped = unwarp(black_line_wraped, Minv, img_size)
    # plt.imshow(black_line_unwraped)
    # plt.title('black_line_unwraped', fontsize=10)
    # mpimg.imsave("black_line_unwraped.png", black_line_unwraped)
    # plt.show()

    road_line = cv2.addWeighted(undist_road, 1., black_line_unwraped, 0.8, 0.)
    # plt.imshow(road_line)
    # plt.show()
    # mpimg.imsave("road_line.png", road_line)#for readme
    return road_line
def visualize_perspective_transfor(undist_road, src):
    # For fun: get perspective transform of the original road image
    # plt.plot(src[0,0],src[0,1],'.')
    # plt.plot(src[1,0],src[1,1],'.')
    # plt.plot(src[2,0],src[2,1],'.')
    #plt.plot(src[3,0],src[3,1],'.')
    #plt.plot(dst[0,0],dst[0,1],'.')
    #plt.plot(dst[1,0],dst[1,1],'.')
    #plt.plot(dst[2,0],dst[2,1],'.')
    #plt.plot(dst[3,0],dst[3,1],'.')
    road_rectangale = cv2.line(undist_road, (src[0,0],src[0,1]), (src[1,0],src[1,1]), (0, 255, 0) , 2)
    road_rectangale=cv2.line(road_rectangale, (src[3,0],src[3,1]), (src[2,0],src[2,1]), (0, 255, 0) , 2)
    road_rectangale=cv2.line(road_rectangale, (src[0,0],src[0,1]), (src[3,0],src[3,1]), (0, 255, 0) , 2)
    road_rectangale=cv2.line(road_rectangale, (src[1,0],src[1,1]), (src[2,0],src[2,1]), (0, 255, 0) , 2)
    return road_rectangale

def Lane_Finding_Pipeline_Image_Advanced(image_road):
    """    Main pipline to detect lane lines"""
    # data = np.load('calib_info.npz')
    # mtx = data['mtx']
    # dist = data['dist']
    # print(mtx)
    # print(dist)
    mtx = np.float32([[1.15777818*10**3, 0.00000000, 6.67113857*10**2],\
    [0.00000000, 1.15282217*10**3, 3.86124583*10**2],\
    [0.0000000, 0.00000000, 1.00000000]])
    dist = np.float32([[-0.24688507, -0.02373155 ,-0.00109831,  0.00035107, -0.00259868]])

    # undist_roadorting the test image_road:
    undist_road = cv2.undistort(image_road, mtx, dist, None, mtx)

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(image_road)
    # ax1.set_title('Original Image', fontsize=10)
    # ax2.imshow(undist_road)
    # ax2.set_title('Undistorted Image', fontsize=10)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # mpimg.imsave("road_undistorted.png", undist_road)# for readme
    # plt.show()

    # Note: img is the undistorted image
    img = np.copy(undist_road)

    sx_binary = grad_thresh(img, thresh=(10,100))#20, 100
    s_binary = colorHSV_thresh(img, thresh=(125,255))
    R_binary = colorBGR_thresh(img, thresh=(200,255))#240,255

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    # color_binary = np.dstack(( np.zeros_like(sx_binary), sx_binary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1) | (R_binary == 1)] = 1

    # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(sx_binary)
    # ax1.set_title('grad thresh binary (sobel x)', fontsize=10)
    # ax2.imshow(s_binary)
    # ax2.set_title('color thresh binary (S from HSV)', fontsize=10)
    # ax3.imshow(R_binary)
    # ax3.set_title('color thresh binary (R from BGR)', fontsize=10)
    # ax4.imshow(combined_binary)
    # ax4.set_title('grad & color combined', fontsize=10)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()

    # Define calibration box in source (original) and destination
    # (desired, warped coordinates)
    img_size = (img.shape[1], img.shape[0])

    # 4 source image points
    src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],#top left
    [((img_size[0] / 6) - 10), img_size[1]],#bottomleft
    [(img_size[0] * 5 / 6) + 45, img_size[1]],# bottom right
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])# top right

    # 4 desired coordinates
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

    # get perspective transform of the binary image
    binary_warped, M, Minv = warp(combined_binary, src, dst, img_size)
    # plt.imshow(binary_warped)
    # plt.title('binary warped (original to pixel)', fontsize=10)
    # plt.show()

    #TODO: write the if condition:
    margin_around_line = 100
    # if not left_fit:
    # Find our lane pixels first
    leftx, lefty, rightx, righty, binary_warped_window,\
    left_lane_inds, right_lane_inds,nonzeroy, nonzerox \
    = find_lane_pixels(binary_warped)

    # plt.imshow(binary_warped_window)
    # plt.title('binary_warped_window', fontsize=10)
    # plt.show()

    binary_warped_window_pixel = visualize_detected_pixels(binary_warped_window, lefty, leftx, righty, rightx)
    # plt.imshow(binary_warped_window_pixel)
    # plt.title('binary_warped_window_pixel', fontsize=10)
    # plt.show()

    # Fit a polynomial
    ploty, left_fitx, right_fitx, left_fit, right_fit \
    = fit_polynomial(leftx, lefty, rightx, righty, binary_warped.shape[0])

    binary_warped_window_pixel_line = draw_line(binary_warped_window_pixel, left_fitx, right_fitx, ploty)
    # plt.imshow(binary_warped_window_pixel_line)
    # plt.title('binary_warped_window_pixel_line', fontsize=10)
    # plt.show()
    # else:
    #     leftx, lefty, rightx, righty, binary_warped_pixel = search_around_poly(binary_warped, left_fit, right_fit, margin_around_line)
    #     # plt.imshow(binary_warped_pixel)
    #     # plt.title('binary warped pixel (search around)', fontsize=10)
    #     # plt.show()
    #     # Fit a polynomial
    #     ploty, left_fitx, right_fitx, left_fit, right_fit = fit_polynomial(binary_warped_pixel, leftx, lefty, rightx, righty, binary_warped.shape[0])
    #     #print(left_fit)
    #     # visualize_region_search_around_poly(binary_warped, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty):
    #     # uuwarped_binary = unwarp(binary_warped_line, Minv, img_size)
    #     # plt.imshow(uuwarped_binary)
    #     # plt.title('unwarped binary', fontsize=10)
    #     # plt.show()

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # calculate the curve raduis in meters
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fitx, right_fitx, ym_per_pix, xm_per_pix)
    #print(left_curverad, 'm', right_curverad, 'm')
    #calculate average of curvature raduis
    R_curve = (left_curverad + right_curverad)/2

    # calculate car offset from center of lane
    car_off_center = measure_off_center_real(left_fitx[0], right_fitx[0], img_size[0],xm_per_pix)

    text_R = '{} meters raduis of curvature'.format(round(R_curve,2))
    if car_off_center >= 0:
        text_C = '{} meters left of center'.format(round(car_off_center,2))
    else:
        text_C = '{} meters right of center'.format(round(-car_off_center,2))

    # Using cv2.putText() method
    # cv2.putText(undist_road, text_C, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #                    1, (255, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(undist_road, text_R, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
    #                   1, (255, 0, 0), 2, cv2.LINE_AA)

    # road_window = visualize_window_serach(binary_warped_window_pixel_line, undist_road,Minv, img_size )
    # road_lines = visualize_lines(undist_road, src, dst, img_size,left_fitx, right_fitx, ploty )

    road_lane = visualize_lane(binary_warped,undist_road, ploty, left_fitx, right_fitx, Minv, img_size)

    ## VISULAIZE for readme:
    # undist_road_temp = np.copy(undist_road)
    # road_rectangale = visualize_perspective_transfor(undist_road_temp, src)
    # plt.imshow(road_rectangale)
    # plt.title('road with rectangle', fontsize=10)
    # mpimg.imsave("road_rectangale.png", road_rectangale)#for readme
    # plt.show()
    # road_rectangale_warped, M, Minv = warp(road_rectangale, src, dst, img_size)
    # plt.imshow(road_rectangale_warped)
    # plt.title('road_rectangale_warped', fontsize=10)
    # mpimg.imsave("road_rectangale_warped.png", road_rectangale_warped)#for readme
    # plt.show()
    # mpimg.imsave("road_undistorted.png", undist_road)
    # mpimg.imsave("sx_binary.png", sx_binary)
    # mpimg.imsave("s_binary.png", s_binary)
    # mpimg.imsave("R_binary.png", R_binary)
    # mpimg.imsave("cmbined_binary.png", combined_binary)
    # mpimg.imsave("binary_warped_window_pixel.png", binary_warped_window_pixel)
    # mpimg.imsave("binary_warped_window_pixel_line.png", binary_warped_window_pixel_line)# for readme
    # mpimg.imsave("road_window.png", road_window)

    return road_lane

"""
cv2.destroyAllWindows()
# Make a list of calibration images
# images = glob.glob('../test_images/straight_lines*.jpg')
images = glob.glob('../test_images/test*.jpg')
# Step through the list and search for chessboard corners
ind_for = 0
for fname in images:
    #print(fname)
    image = cv2.imread(fname)

    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)

    result = Lane_Finding_Pipeline_Image_Advanced(image)
    plt.imshow(result)
    plt.show()
    #mpimg.imsave('{:03d}.png'.format(fname), result)#for readme
    mpimg.imsave("frame" + str(ind_for) + ".png", result)
    #"file_" + str(i) + ".dat", 'w', encoding='utf-8'
    cv2.waitKey(500) # waits until a key is pressed
    ind_for = ind_for +1

cv2.destroyAllWindows()# destroys the window showing image


image_path = '../test_images/straight_lines1.jpg'
image = cv2.imread(image_path)
road_lane = Lane_Finding_Pipeline_Image_Advanced(image)
# plt.imshow(road_lane)
# plt.show()
# mpimg.imsave("road_lane.png", road_lane)
"""

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

video_out = 'project_video_output.mp4'
video_out = 'challenge_video_output.mp4'
video_out = 'harder_challenge_video.mp4'

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

clip1 = VideoFileClip("../project_video.mp4")#.subclip(0,2)
clip1 = VideoFileClip("../challenge_video.mp4")#.subclip(0,2)
clip1 = VideoFileClip("../harder_challenge_video.mp4")#.subclip(0,2)
clip = clip1.fl_image(Lane_Finding_Pipeline_Image_Advanced) #NOTE: this function expects color images!!
clip.write_videofile(video_out, audio=False)
