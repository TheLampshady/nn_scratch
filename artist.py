import numpy
import scipy.ndimage
import matplotlib.pyplot

from utils import to_nn_input


def draw_number(number_row):
    image_array = numpy.asfarray(number_row[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")


def draw_dream(number_row):
    image_array = numpy.asfarray(number_row).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")


def rotate_number(img, degree=10):
    arr_img = to_nn_input(img)
    result = scipy.ndimage.interpolation.rotate(
        arr_img.reshape(28,28),
        degree,
        cval=0.01,
        reshape=False
    )
    return [0] + list(result.reshape(784))
