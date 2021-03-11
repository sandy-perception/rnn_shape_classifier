import numpy as np
import cv2
import os
import shutil
import random

from matplotlib import pyplot as plt

color = (255, 255, 255)
thick = 2
im_width = 104
im_height = 64
im_shape=(im_height,im_width)


def get_blank_im():
    """
    Create blank image

    :return: blank image with defined im_shape
    """
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


def save_incremental_shape_imgs(shape_dir,vect_instructions,shape_id):

	vecs = vect_instructions.split(";")[:-1]

	im = get_blank_im()
	tuples = []
	for vec in vecs :
		tuples.append(vec_to_tuple(vec))

	start = tuples[0]
	paths=[]
	for j in range(1 ,len(tuples)):
		end = (start[0] + tuples[j][0] ,start[1 ] +tuples[j][1])
		im = cv2.line(im ,start ,end ,color ,thick)
		f_name=shape_id+"_"+str(j)+".png"
		path=os.path.join(shape_dir,f_name)
		cv2.imwrite(path,im)
		paths.append(path)
		start = end

	return paths


def generate_intermediate_images(category_lines,cols=5):
	temp = "temp"
	if os.path.exists(temp):
		shutil.rmtree(temp)

	os.makedirs(temp)

	shapes_list = []
	category_list = []
	for category, samples in category_lines.items():
		print("Generating images for :",category)
		category_list.append(category)
		shape_dir = os.path.join(temp, category)
		os.makedirs(shape_dir)
		j = 0
		path_list = []
		while j < cols:
			sample = random.choice(samples)
			shape_id = category + "_" + str(j)
			paths = save_incremental_shape_imgs(shape_dir, sample, shape_id)
			path_list.append(paths)
			j+=1

		shapes_list.append(path_list)

	return shapes_list,category_list

def save_collages(shapes_list,cols,category_list):

	temp = "temp_gif"
	if os.path.exists(temp):
		shutil.rmtree(temp)

	os.makedirs(temp)

	rows = len(shapes_list)
	grid_indices = np.zeros((rows,cols),dtype=int)
	max_indices = np.zeros((rows, cols), dtype=int)

	for i in range(rows):
		path_list = shapes_list[i]
		for j in range(cols):
			max_indices[i,j] = len(path_list[j])
	index=0
	while not np.array_equal(max_indices,grid_indices) :
		print("max_indices array ")
		print(max_indices)
		print("grid_indices array ")
		print(grid_indices)
		fig = plt.figure(figsize=(cols * 4, rows * 3))
		for i in range(rows):
			for j in range(cols):
				if grid_indices[i,j] < max_indices[i,j] :
				   grid_indices[i,j] += 1

				path_index = grid_indices[i,j] - 1
				im_index = i * cols + j + 1
				im_path = shapes_list[i][j][path_index]
				print("im_path :", im_path)
				im = cv2.imread(im_path)
				ax = plt.subplot(rows, cols, im_index)
				ax.set_title(category_list[i])
				ax.imshow(im)

		path=os.path.join(temp,str(index)+".png")
		index +=1
		fig.savefig(path)  # save the figure to file
		plt.close(fig)



def get_sample_gif(category_lines,cols=5):

	shapes_list,category_list = generate_intermediate_images(category_lines,cols)
	save_collages(shapes_list,cols,category_list)
