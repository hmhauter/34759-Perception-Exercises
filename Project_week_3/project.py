import cv2
from matplotlib import pyplot as plt
import numpy as np

def calculate_sum_differences(img1, img2):
    img1 = np.array(img1, dtype=np.int32)
    img2 = np.array(img2, dtype=np.int32)
    return np.sum(np.abs(img1-img2))

def show_two_images(img1, img2):
    plt.gray()
    f, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18,18))
    ax_left.imshow(img1)
    ax_right.imshow(img2)    
    plt.show()

def match_span(span, template):
    sz = template.shape[1]
    diff_storage = []
    for x in range(span.shape[1]-sz):
        diff = calculate_sum_differences(span[:, x:x+sz], template)
        diff_storage.append(diff)
    # find min difference
    diff_storage = np.array(diff_storage)
    min = diff_storage.argmin()
    return min 

def show_match(img, template, min_value):
    print(min_value)
    f, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18,18))
    bottom_right = (min_value + template.shape[1], 0 + template.shape[0])
    match = cv2.rectangle(img.copy(), (min_value,0), bottom_right, (0,0,255), 3)
    ax_left.imshow(match)
    ax_right.imshow(template)
    plt.show()

def template_matching(img1, img2, sz):
    diff_storage = np.zeros_like(img1)
    for r in range(img1.shape[0]-sz):
        for c in range(img1.shape[1]-sz):
            template = img1[r:r+sz, c:c+sz]
            min = match_span(img2[r:r+sz, :], template)   
            diff = abs(min - c)         
            diff_storage[r][c] = diff
    print(diff_storage)
    return diff_storage
    
def validate(img1, img2):
    method = cv2.TM_SQDIFF
    res = cv2.matchTemplate(img1,img2,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return min_loc[0]


_dir = '/home/hhauter/Documents/S23/Perception/34759-Perception-Exercises/Project_week_3/'
nose_left = cv2.imread(_dir+"nose_left.png")
nose_right = cv2.imread(_dir+"nose_right.png")
nose_1 = cv2.imread(_dir+"nose1.png")
nose_2 = cv2.imread(_dir+"nose2.png")
nose_3 = cv2.imread(_dir+"nose3.png")
nose_span = cv2.imread(_dir+"nose_span.png")
nose_left_gray = cv2.cvtColor(nose_left, cv2.COLOR_BGR2GRAY)
nose_right_gray = cv2.cvtColor(nose_right, cv2.COLOR_BGR2GRAY)
nose_1_gray = cv2.cvtColor(nose_1, cv2.COLOR_BGR2GRAY)
nose_2_gray = cv2.cvtColor(nose_2, cv2.COLOR_BGR2GRAY)
nose_3_gray = cv2.cvtColor(nose_3, cv2.COLOR_BGR2GRAY)
nose_span_gray = cv2.cvtColor(nose_span, cv2.COLOR_BGR2GRAY)

diff = calculate_sum_differences(nose_left_gray, nose_right_gray)
print(diff)

diff = calculate_sum_differences(nose_left_gray, nose_1_gray)
print(diff)
diff = calculate_sum_differences(nose_left_gray, nose_2_gray)
print(diff)
diff = calculate_sum_differences(nose_left_gray, nose_3_gray)
print(diff)

# show_two_images(nose_left_gray, nose_2_gray)

min = match_span(nose_span_gray, nose_2_gray)
show_match(nose_span_gray, nose_2_gray, min)

min = validate(nose_span_gray, nose_2_gray)
print(min)

tsukuba_left = cv2.imread(_dir+"tsukuba_left.png")
tsukuba_right = cv2.imread(_dir+"tsukuba_right.png")
tsukuba_left_gray = cv2.cvtColor(tsukuba_left, cv2.COLOR_BGR2GRAY)
tsukuba_right_gray = cv2.cvtColor(tsukuba_right, cv2.COLOR_BGR2GRAY)

# rescale images (can be left out)
scaler = 3
img_size = (int(tsukuba_left_gray.shape[1]/scaler), int(tsukuba_left_gray.shape[0]/scaler))
tsukuba_left_gray = cv2.resize(tsukuba_left_gray, img_size, interpolation=cv2.INTER_AREA)
tsukuba_right_gray = cv2.resize(tsukuba_right_gray, img_size, interpolation=cv2.INTER_AREA)


disp = template_matching(tsukuba_left_gray, tsukuba_right_gray, 7)
disp = np.array(disp,dtype=np.float64)
disp *= (255.0/disp.max())
print(disp)
plt.gray()
plt.figure(figsize=(18,18))
plt.imshow(disp)
plt.show()

# Initialize the stereo block matching object 
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
# Compute the disparity image
disparity = stereo.compute(tsukuba_left_gray, tsukuba_right_gray).astype(np.float32) / 16.0
plt.figure(figsize=(18,18))
plt.gray()
plt.imshow(disparity)
plt.show()


f, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18,18))
ax_left.imshow(disp)
ax_right.imshow(disparity)
plt.gray()
plt.show()

# further improvements: 
# optimize code (no for looping in calculate_sum_differences)
# early stopping: disparity -> things can not be located to the right (finger jumps to the left)

