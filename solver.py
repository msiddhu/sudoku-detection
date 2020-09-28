import argparse
import re

import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# construct the argument parser and parse the arguments
from extractcell import extractDigit
from findpuzzle import findPuzzle
from backtrack import sudosolve

"""
cmd:    python solver.py --model output --image image\img2.jpg --debug 1



ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
                help="path to trained digit classifier")
ap.add_argument("-i", "--image",
                help="path to input Sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())


# solution.show_full()
# loop over the cell locations and board
"""


def getsolution(file_path) :
    model = load_model('scnn.h5')
    # load the input image from disk and resize it

    # image = cv2.imread(args["image"])

    # image = cv2.imread('images/img6.jpeg')
    image = cv2.imread(file_path)
    image = imutils.resize(image, width=600)
    # find the puzzle in the image and then
    (puzzleImage, warped) = findPuzzle(image, debug=0)
    # initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int")
    # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []
    for y in range(0, 9) :
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9) :
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY :endY, startX :endX]

            digit = extractDigit(cell, x, y, debug=0)
            # plt.imsave('{}{}.jpg'.format(startX, endY), digit)
            # verify that the digit is not empty
            if digit is not None :
                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                # classify the digit and update the Sudoku board with the
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]
                # print('{}-{} pred={}'.format(y, x, pred))
                board[y, x] = pred
        # add the row to our cell locations
        cellLocs.append(row)
    # construct a Sudoku puzzle from the board

    # puzzle.show()
    # solve the Sudoku puzzle
    board = sudosolve(board)
    print(board)
    return np.array_str(board)
