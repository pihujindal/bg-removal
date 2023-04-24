import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

# loading the mountain image
mountain = cv2.imread('mt.jpg')

# resizing the mountain image as 640 X 480
resized_image = cv2.resize(mountain, (640, 480))

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # create thresholds
        lower_bound = np.array([100, 100, 100])
        upper_bound = np.array([255, 255, 255])

        # thresholding image
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # inverting the mask
        inverted_mask = cv2.bitwise_not(mask)

        # bitwise and operation to extract foreground / person
        foreground = cv2.bitwise_and(frame, frame, mask=inverted_mask)

        # set the background to the mountain image
        background = resized_image.copy()

        # perform bitwise and operation to get the final image
        final_image = cv2.bitwise_and(background, background, mask=mask)
        final_image = cv2.add(final_image, foreground)

        # show the final image
        cv2.imshow('frame', final_image)

        # wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
