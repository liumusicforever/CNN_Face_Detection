import matplotlib.pyplot as plt
import cv2

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


img_path = '/home/share/data/FDDB/2002/07/19/big/img_130.jpg'
img = cv2.imread(img_path)


pyramid_t = 3
win_size = (48,48)
win_stride =  10


# Generate Gaussian pyramid for img
imgPyramids = [img.copy()]
for i in range(1, pyramid_t):
    imgPyramids.append(cv2.pyrDown(imgPyramids[i - 1]))
for i in range(pyramid_t):
    image = imgPyramids[i]
    for (x, y, window) in sliding_window(image, stepSize=win_stride, windowSize=win_size):
		# if the window does not meet our desired window size, ignore it
        if window.shape[0] != win_size[0] or window.shape[1] != win_size[1]:
            continue

        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + win_size[0], y + win_size[1]), (255, 0, 0), 2)
        face = image[y : y+win_size[1] , x : x+win_size[0]]
        plt.imshow(clone)
        plt.show(block = False)
        plt.pause(0.1)