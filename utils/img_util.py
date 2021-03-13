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
    """
    convert vector string to tuple of integers
    :param vec: tuple with integers
    :return: tuple of 2 integers
    """
    digits = vec[1:-1].split(",")
    return (int(digits[0]),int(digits[1]))


def get_shape_img(vect_instructions):
    """
    Create empty image and draw lines according to vectors instructions
    :param vect_instructions: list containing start coordinate of shape
    and consequent path coordinates to construct shape
    :return: 2d image with shape drawn according to instructions
    """
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
    """
    Create sequence of images as if these are frames of video and in which
    shape get constructed according to given vector sequence.

    :param shape_dir: directory where images get saved
    :param vect_instructions: string having starting point of shape construct and subsequent vectors
    :param shape_id: id of shape
    :return: list of paths of saved images
    """
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
    """
    Call save_incremental_shape_imgs for multiple shapes
    :param category_lines: dict containing shape name and vector instructions
    :param cols: number of columns in collage
    :return: list of paths of all shapes, list of categories
    """
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
    """
    Create and save collages of shape images. The whole set of collages will
    show that each shape get constructed sequentially.

    :param shapes_list: list containing file path of each shape
    :param cols: number of columns in collage
    :param category_list: list of classes
    :return: path where collages get saved
    """
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

    return temp


def generate_gif(path):
    """
    Generate gif file from given images

    :param path: path containing set of images
    :return: gif file
    """
    pass


def get_sample_gif(category_lines,cols=5):
    """
    High level method to generate shape collages which will
    have construction state of all shapes
    :param category_lines: dict containing shape name and their vector instruction set
    :param cols: number of columns in collage
    :return:
    """
    shapes_list,category_list = generate_intermediate_images(category_lines,cols)
    collage_path = save_collages(shapes_list,cols,category_list)
    return generate_gif(collage_path)
