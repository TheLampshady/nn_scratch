import numpy
import scipy.ndimage
import matplotlib.pyplot


def draw_number(number_row):
    image_array = numpy.asfarray(number_row[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")


def draw_dream(number_row):
    image_array = numpy.asfarray(number_row).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")


def rotate_number(img, degree=10):
    return scipy.ndimage.interpolation.rotate(
        img.reshape(28,28),
        degree,
        cval=0.01,
        reshape=False
    )
