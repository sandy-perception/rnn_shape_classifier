import numpy as np
import cv2

color = (255, 255, 255)
thick = 2
im_width = 104
im_height = 64
im_shape=(im_height,im_width)


def get_blank_im():
    global im_shape
    return np.zeros(im_shape,dtype=np.int8)


def vec_to_tuple(vec):
    digits = vec[1:-1].split(",")
    return (int(digits[0]),int(digits[1]))


def get_shape_img(vect_instructions):

	vecs = vect_instructions.split(";")[:-1]

	im = get_blank_im()
	tuples = []
	for vec in vecs :
		tuples.append(vec_to_tuple(vec))

	start = tuples[0]
	for j in range(1 ,len(tuples)):
		end = (start[0] + tuples[j][0] ,start[1 ] +tuples[j][1])
		im = cv2.line(im ,start ,end ,color ,thick)
		start = end

	return im