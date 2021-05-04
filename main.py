from cv2 import imread,imwrite,imshow,waitKey,IMREAD_GRAYSCALE
import os, sys
import cv2
import numpy as np

background_white = 225
background_green = 80
max_bbox = 100

scale = 8
scale_factor = 1.2
fast_n = 9
nkeypoints = 4000
fast_threshold = 0.08
patch_size = 31


def hamming_distance(x, y):
    return np.count_nonzero(x[:, None, :] != y[None, :, :], axis=-1).astype(np.float32)


# Do image convolution
def convolution(v_img, v_kernel):
    kernel_h = v_kernel.shape[0]
    kernel_w = v_kernel.shape[1]

    # Pad the border
    image_pad = np.pad(v_img, pad_width=(
        (kernel_h // 2, kernel_h // 2), (kernel_w // 2,
                                         kernel_w // 2)), mode='constant', constant_values=0).astype(np.float32)

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image_pad.shape)

    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            # sum = 0
            x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * v_kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w

    if h == 0:
        return image_conv[h:, w:w_end]
    if w == 0:
        return image_conv[h:h_end, w:]
    return image_conv[h:h_end, w:w_end]


# Detect the FAST corner points
# Return in (x,y)
def FAST(img, N=9, threshold=0.15, nms_window=2):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16  # 3x3 Gaussian Window

    img = convolution(img, kernel)

    cross_idx = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
    circle_idx = np.array([[3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
                           [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]])

    corner_img = np.zeros(img.shape)
    keypoints = []
    for y in range(3, img.shape[0] - 3):
        for x in range(3, img.shape[1] - 3):
            Ip = img[y, x]
            t = threshold * Ip if threshold < 1 else threshold
            # Fast checking key idx only
            if np.count_nonzero(Ip + t < img[y + cross_idx[0, :], x + cross_idx[1, :]]) >= 3 or np.count_nonzero(
                    Ip - t > img[y + cross_idx[0, :], x + cross_idx[1, :]]) >= 3:
                # Detailed check
                if np.count_nonzero(img[y + circle_idx[0, :], x + circle_idx[1, :]] >= Ip + t) >= N or np.count_nonzero(
                        img[y + circle_idx[0, :], x + circle_idx[1, :]] <= Ip - t) >= N:
                    keypoints.append([x, y])
                    corner_img[y, x] = np.sum(np.abs(Ip - img[y + circle_idx[0, :], x + circle_idx[1, :]]))

    # NMS - Non Maximal Suppression
    if nms_window != 0:
        fewer_kps = []
        for [x, y] in keypoints:
            # Ensure the range of the window
            window = corner_img[
                     np.clip(y - nms_window, 0, corner_img.shape[0]):np.clip(y + nms_window + 1, 0,
                                                                             corner_img.shape[0]),
                     np.clip(x - nms_window, 0, corner_img.shape[1]):np.clip(x + nms_window + 1, 0,
                                                                             corner_img.shape[0])]
            loc_y_x = np.unravel_index(window.argmax(), window.shape)
            x_new = x + loc_y_x[1] - nms_window
            y_new = y + loc_y_x[0] - nms_window
            new_kp = [x_new, y_new]
            if new_kp not in fewer_kps:
                fewer_kps.append(new_kp)
    else:
        fewer_kps = keypoints

    return np.array(fewer_kps), corner_img


# Compute the orientation of each keypoint
def corner_orientations(img, corners):
    OFAST_MASK = np.zeros((31, 31), dtype=np.int32)
    OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
    for i in range(-15, 16):
        for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
            OFAST_MASK[15 + j, 15 + i] = 1
    mrows, mcols = OFAST_MASK.shape
    mrows2 = int((mrows - 1) / 2)
    mcols2 = int((mcols - 1) / 2)

    # Padding to avoid errors at corners near image edges.
    img = np.pad(img, (mrows2, mcols2), mode='constant', constant_values=0)

    # Calculating orientation by the intensity centroid method
    orientations = []
    for i in range(corners.shape[0]):
        c0, r0 = corners[i, :]
        m01, m10 = 0, 0
        for r in range(mrows):
            m01_temp = 0
            for c in range(mcols):
                if OFAST_MASK[r, c]:
                    I = img[r0 + r, c0 + c]
                    m10 = m10 + I * (c - mcols2)
                    m01_temp = m01_temp + I
            m01 = m01 + m01_temp * (r - mrows2)
        orientations.append(np.arctan2(m01, m10))
    return np.array(orientations)


# Compute rBRIEF descriptor
def rBRIEF(img, keypoints, orientations, n=256, patch_size=9):
    random = np.random.RandomState(seed=42)

    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]]) / 273  # 5x5 Gaussian Window

    img = convolution(img, kernel)

    # Uniform sample
    samples = random.randint(-(patch_size - 2) // 2 + 1, (patch_size // 2), (n * 2, 2))
    samples = np.array(samples, dtype=np.int32)
    pos1, pos2 = np.split(samples, 2)

    rows, cols = img.shape

    # Drop the keypoints near the border
    distance = int((patch_size // 2) * 1.5)
    mask = (((distance - 1) < keypoints[:, 0])
            & (keypoints[:, 0] < (cols - distance + 1))
            & ((distance - 1) < keypoints[:, 1])
            & (keypoints[:, 1] < (rows - distance + 1)))

    keypoints = np.array(keypoints[mask], dtype=np.intp, copy=False)
    orientations = np.array(orientations[mask], copy=False)
    descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

    # Compute the descriptor
    for i in range(descriptors.shape[0]):
        angle = orientations[i]
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)

        kr = keypoints[i, 1]
        kc = keypoints[i, 0]
        for p in range(pos1.shape[0]):
            pr0 = pos1[p, 0]
            pc0 = pos1[p, 1]
            pr1 = pos2[p, 0]
            pc1 = pos2[p, 1]

            # Rotation is based on the idea that:
            # x` = x*cos(th) - y*sin(th)
            # y` = x*sin(th) + y*cos(th)
            # c -> x & r -> y
            spr0 = round(sin_theta * pr0 + cos_theta * pc0)
            spc0 = round(cos_theta * pr0 - sin_theta * pc0)
            spr1 = round(sin_theta * pr1 + cos_theta * pc1)
            spc1 = round(cos_theta * pr1 - sin_theta * pc1)

            if img[int(kr + spr0), int(kc + spc0)] < img[int(kr + spr1), int(kc + spc1)]:
                descriptors[i, p] = True
    return keypoints, descriptors


def resize(v_img, v_shape):
    origin_shape = v_img.shape[0]
    factor = origin_shape / v_shape
    o_img = np.zeros((v_shape,v_shape)).astype(v_img.dtype)
    for y in range(v_shape):
        for x in range(v_shape):
            o_img[y, x] = v_img[int(y * factor), int(x * factor)]
    return o_img


def gaussian_blur(v_img, v_factor):
    filter_size = 2 * int(4 * v_factor + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (v_factor ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * v_factor ** 2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2

    o_img = np.zeros_like(v_img, dtype=np.float32)
    o_img[:, :] = convolution(v_img[:, :], gaussian_filter)
    return o_img.astype(np.uint8)


def match_keypoints(descriptors1, descriptors2, max_distance=np.inf, cross_check=True, distance_ratio=None):
    distances = hamming_distance(descriptors1, descriptors2)
    indices1 = np.arange(descriptors1.shape[0])  # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)] "indices of d1"
    indices2 = np.argmin(distances,
                         axis=1)  # [12, 465, 23, 111, 123, 45, 67, 2, 265, ..., len(d1)] "list of the indices of d2 points that are closest to d1 points"
    # Each d1 point has a d2 point that is the most close to it.
    if cross_check:
        matches1 = np.argmin(distances, axis=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if distance_ratio is not None:
        modified_dist = distances
        fc = np.min(modified_dist[indices1, :], axis=1)
        modified_dist[indices1, indices2] = np.inf
        fs = np.min(modified_dist[indices1, :], axis=1)
        mask = fc / fs <= distance_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    # sort matches using distances
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches


# def detect_surf_features_cv(v_img):
#     # sift = cv2.BRISK_create(octaves=8)
#     orb = cv2.ORB_create(nfeatures=2000, WTA_K=4)
#     # kp=sift.detect(v_img)
#     # kp,des=sift.compute(v_img,kp,)
#     kp, des = orb.detectAndCompute(v_img, None)
#     return kp, des

#
# def detect_surf_features_SCIPY(v_img):
#     orb = ORB(n_keypoints=2000)
#     orb.detect_and_extract(cv2.cvtColor(v_img, cv2.COLOR_BGR2GRAY))
#     kp = orb.keypoints
#     kp = [cv2.KeyPoint(item[1], item[0], 1) for item in kp]
#     return kp, orb.descriptors
#

def detect_orb_features_ours(v_img):
    pyramids = [v_img]

    for i in range(1, scale - 1):
        blurred = gaussian_blur(pyramids[i - 1], v_factor=scale_factor)
        blurred_resized = resize(blurred,
                             int(blurred.shape[0] / (scale_factor)))
        pyramids.append(blurred_resized)

    keypoints_list = []
    descriptors_list = []
    response_list = []
    for octave in range(len(pyramids)):
        octave_image = pyramids[octave]

        # Detect
        keypoints, corner_img = FAST(octave_image, fast_n,
                                     fast_threshold, nms_window=4)
        orientations = corner_orientations(octave_image, keypoints)

        keypoints, descriptors = rBRIEF(octave_image, keypoints, orientations, patch_size=patch_size)
        response_list.append(corner_img[keypoints[:, 1], keypoints[:, 0]])
        scaled_keypoints = keypoints * scale_factor ** octave
        keypoints_list.append(scaled_keypoints)
        descriptors_list.append(descriptors)

    keypoints = np.vstack(keypoints_list)
    descriptors = np.vstack(descriptors_list).view(bool)
    responses = np.concatenate(response_list)
    if len(keypoints) > nkeypoints:
        best_indices = responses.argsort()[::-1][:nkeypoints]
        keypoints = keypoints[best_indices]
        descriptors = descriptors[best_indices]

    # return [cv2.KeyPoint(item[0], item[1], 1) for item in keypoints], np.array(descriptors)
    return keypoints, np.array(descriptors)


def isGoodTransform(v_trans, v_p1, v_p2):
    EPS = 4
    p1 = np.ones((3, 1), dtype=np.float32)
    p2 = np.ones((3, 1), dtype=np.float32)
    p1[0][0] = v_p1[0]
    p1[1][0] = v_p1[1]
    p2[0][0] = v_p2[0]
    p2[1][0] = v_p2[1]
    p_ = np.matmul(v_trans, p1)
    distance = np.abs(p_ - p2)
    if (distance[0][0] + distance[1][0]) < EPS:
        return True
    return False

def random3indices(num_points):
    res = []
    for i in range(3):
        idx = np.random.randint(0, high=num_points)
        if idx not in res:
            res.append(idx)
        else:
            flag = True
            while(flag):
                idx = np.random.randint(0, high=num_points)
                if idx not in res:
                    flag = False
                    res.append(idx)
    return res

def ransac_ours(v_point1,v_point2):
    num_points = v_point1.shape[0]
    transform=np.zeros((2,3)).astype(np.float32)
    mask=np.zeros((num_points, 1), dtype='uint8')
    ran_iters = 10000
    satisfy_points = 0
    for iter in range(ran_iters):
        # random 3 indices of points
        # ran_indices = rng.choice(num_points, 3, replace=False)
        ran_indices = random3indices(num_points)
        # solve the transform matrix given these 3 points
        P = np.ones((3, 3), dtype=np.float32)
        P_ = np.ones((3, 3), dtype=np.float32)
        for i in range(3):
            P[0][i] = v_point1[ran_indices[i]][0]
            P[1][i] = v_point1[ran_indices[i]][1]
            P_[0][i] = v_point2[ran_indices[i]][0]
            P_[1][i] = v_point2[ran_indices[i]][1]
        trans = np.matmul(P_, np.linalg.inv(P))
        # apply this transformation to each point
        num_satisfy = 0
        mask_satisfy = np.zeros((num_points, 1), dtype='uint8')
        for p in range(num_points):
            if isGoodTransform(trans, v_point1[p], v_point2[p]):
                num_satisfy += 1
                mask_satisfy[p] = 1
        # if better, update the best transform matrix
        if num_satisfy > satisfy_points:
            satisfy_points = num_satisfy
            transform = trans[:2]
            mask = mask_satisfy

    return transform,mask

def warpAffine(v_img,v_tr):
    h, w = v_img.shape[:2]
    dst_y, dst_x = np.indices((h, w))
    dst_lin_homg_pts = np.stack((dst_x.ravel(), dst_y.ravel(), np.ones(dst_y.size)))

    v_tr_inv=np.zeros((3,3))
    v_tr_inv[:2]=v_tr
    v_tr_inv[2,2]=1
    v_tr_inv=np.linalg.inv(v_tr_inv)[:2]
    src_lin_pts = np.round(v_tr_inv.dot(dst_lin_homg_pts)).astype(int)

    valid_index=np.logical_and(src_lin_pts[0]>0,src_lin_pts[1]>0)
    valid_index=np.logical_and(valid_index,src_lin_pts[0]<h)
    valid_index=np.logical_and(valid_index,src_lin_pts[1]<w)
    src = np.zeros_like(v_img, dtype=np.uint8)

    src_mask=src_lin_pts[:,valid_index]
    dst_mask=dst_lin_homg_pts[:2,valid_index].astype(np.int32)

    src[dst_mask[1], dst_mask[0]] = v_img[src_mask[1],src_mask[0]]
    return src

if __name__ == "__main__":
    if len(sys.argv) !=3:
        print("Usage: python main.py img_path1 img_path2")
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    print(img1_path)
    print(img2_path)
    for item_fast_n, item_fast_threshold, item_patch_size in [
        (11, 0.08, 31),
    ]:
        fast_n = item_fast_n
        fast_threshold = item_fast_threshold
        patch_size = item_patch_size

        # cv2.imshow("", mask)
        # cv2.waitKey()

        # Load imgs
        test_imgs = [
            imread(img1_path,IMREAD_GRAYSCALE),
            imread(img2_path,IMREAD_GRAYSCALE)
        ]

        # test_imgs = []
        # for i in range(1, 6):
            # img = imread("cp_data/images/{:03d}.png".format(i),IMREAD_GRAYSCALE)
            # test_imgs.append(img)

        # Generate pairs
        img_pairs = []
        for i in range(len(test_imgs)):
            for j in range(i + 1, len(test_imgs)):
                img_pairs.append((test_imgs[i], test_imgs[j]))

        for id_img in range(0, len(img_pairs)):
            img1, img2 = img_pairs[id_img]
            try:
                # kp1, des1 = detect_surf_features_cv(img1)
                # kp2, des2 = detect_surf_features_cv(img2)
                # kp1, des1 = detect_surf_features_SCIPY(img1)
                # kp2, des2 = detect_surf_features_SCIPY(img2)
                print("===================={} th==============".format(id_img))
                print("Detect features for the first image")
                kp1, des1 = detect_orb_features_ours(img1)
                print("Detect features for the second image")
                kp2, des2 = detect_orb_features_ours(img2)

                print("Match keypoints")
                matches = match_keypoints(des1, des2, distance_ratio=0.75)
                # matches = [cv2.DMatch(item[0], item[1], 0) for item in matches]
                # matches = [cv2.DMatch(item[0], item[1], 0) for item in matches]

                if len(matches)<3:
                    print("Less than 3 match, can not match")
                    continue

                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                for i, match in enumerate(matches):
                    # points1[i, :] = kp1[match.queryIdx].pt
                    points1[i, :] = kp1[match[0]]
                    points2[i, :] = kp2[match[1]]
                    # points2[i, :] = kp2[match.trainIdx].pt

                # homography, homography_mask = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC,
                                                                        #   maxIters=10000,
                                                                        #   confidence=0.95)
                print("Find transform")
                homography, homography_mask = ransac_ours(points1, points2)

                height, width = img2.shape

                #
                # Warp the img and calculate mask2
                #
                # im1Reg = cv2.warpAffine(img1, homography, (width, height))
                im1Reg = warpAffine(img1, homography)
                valid_points2 = points2[np.logical_and(homography_mask[:, 0], True)]
                center_x = valid_points2[:, 0].mean()
                center_y = valid_points2[:, 1].mean()

                mask2 = np.zeros_like(img2)
                mask2[np.abs(np.float32(im1Reg) - np.float32(img2)) < 30] = 255
                mask2[np.abs(np.float32(im1Reg) - np.float32(background_white)) < 20] = 0
                mask2[np.abs(np.float32(img2) - np.float32(background_white)) < 20] = 0
                mask2[np.abs(np.float32(img2) - np.float32(background_green)) < 20] = 0
                mask2[np.abs(np.float32(im1Reg) - np.float32(background_green)) < 20] = 0
                mask2[np.abs(np.float32(im1Reg) - 255) < 5] = 0
                mask2[np.abs(np.float32(img2) - 255) < 5] = 0
                mask2[im1Reg <= 5] = 0
                for y in range(mask2.shape[0]):
                    for x in range(mask2.shape[1]):
                        if (y - center_y) ** 2 + (x - center_x) ** 2 > max_bbox ** 2:
                            mask2[y, x] = 0

                #
                # Do the same to img1
                #
                height, width = img1.shape
                # homography, homography_mask = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC,
                                                                        #   maxIters=10000,
                                                                        #   confidence=0.95)
                print("Find transform again")
                # homography, homography_mask = ransac_ours(points1, points2)
                homography_inv=np.zeros((3,3))
                homography_inv[:2]=homography
                homography_inv[2,2]=1
                homography_inv=np.linalg.inv(homography_inv)[:2]
                homography=homography_inv

                # im2Reg = cv2.warpAffine(img2, homography, (width, height))
                im2Reg = warpAffine(img2, homography)
                valid_points1 = points1[np.logical_and(homography_mask[:, 0], True)]
                center_x = valid_points1[:, 0].mean()
                center_y = valid_points1[:, 1].mean()

                # delta_img[im1Reg==(0,0,0)]=0
                mask1 = np.zeros_like(img1)
                mask1[np.abs(np.float32(im2Reg) - np.float32(img1)) < 30] = 255
                mask1[np.abs(np.float32(im2Reg) - np.float32(background_white)) < 20] = 0
                mask1[np.abs(np.float32(img1) - np.float32(background_white)) < 20] = 0
                mask1[np.abs(np.float32(img1) - np.float32(background_green)) < 20] = 0
                mask1[np.abs(np.float32(im2Reg) - np.float32(background_green)) < 20] = 0
                mask1[np.abs(np.float32(im2Reg) - 255) < 5] = 0
                mask1[np.abs(np.float32(img1) - 255) < 5] = 0
                mask1[im2Reg < 5] = 0
                for y in range(mask1.shape[0]):
                    for x in range(mask1.shape[1]):
                        if (y - center_y) ** 2 + (x - center_x) ** 2 > max_bbox ** 2:
                            mask1[y, x] = 0

                # img4 = cv2.drawKeypoints(img1, kp1, None)
                # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                #                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                # cv2.imshow("", np.hstack([img3, img4]))
                # cv2.waitKey()
                # imshow("", np.vstack([
                    # np.hstack([img1, im2Reg, mask1]),
                    # np.hstack([im1Reg, img2, mask2])
                # ]))
                # waitKey()
                print("Write image")
                imwrite("{}_{}_{}_{}.png".format(
                    item_fast_n,
                    item_fast_threshold,
                    item_patch_size,
                    id_img,
                ), np.vstack([
                    np.hstack([img1, im1Reg, mask1]),
                    np.hstack([img2, im2Reg, mask2]),
                ]))
                imwrite("1_mask.png",mask1)
                imwrite("2_mask.png",mask2)
                print("Output 1_mask.png, 2_mask.png. The rest is for comparison.")
                pass
            except Exception as e:
                print(e)
                print("Can not match")

        pass
