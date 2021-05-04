# Usage

python main.py img_path1 img_path2

`img_path1` and `img_path2` are the path of the input images

# Note

- Only use the cv2 for image loading/saving (imread,imwrite,imshow,waitKey,IMREAD_GRAYSCALE).
- Manually implement detecting orb features, matching keypoints, finding transform using ransac, warping transform and the matching.
- Result is store in `1_mask.png`, `2_mask.png`. We also output the transformed image for comparison.
- Results on both `images` and `cards` set are also included in the folder. 

