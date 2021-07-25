import numpy as np
import cv2

image = cv2.imread("1.jpg")
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dst = cv2.Canny(grayImage, 0, 150)
cv2.imwrite("canny.jpg", dst)

lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 60, 20)

lines_x = []
# Get height and width to constrain detected lines
height, width, channels = image.shape
for i in range(0, len(lines)):
    l = lines[i][0]
    # Check if the lines are vertical or not
    angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180.0 / np.pi
    if (l[2] > width / 4) and (l[0] > width / 4) and (70 < angle < 100):
        lines_x.append(l[2])
        # To draw the detected lines
        #cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

#cv2.imwrite("lines_found.jpg", image)
# Sorting to get the line with the maximum x-coordinate for proper cropping
lines_x.sort(reverse=True)
crop_image = "cropped_lines"
for i in range(0, len(lines_x)):
    if i == 0:
        # Cropping to the end
        img = image[0:height, lines_x[i]:width]
    else:
        # Cropping from the start
        img = image[0:height, 0:lines_x[i]]
    cv2.imwrite(crop_image + str(i) + ".jpg", img)
