import cv2 as cv
import sys
import os
from numpy import dot, array, flip
import numpy as np
from numpy.linalg import norm
from copy import deepcopy
import argparse

# Color gradient
class ColorGradient:
    def __init__(self,first_color=array([0,0,0]),second_color=array([1,1,1]),start_point=array([0,0]),end_point=array([1,1])) -> None:
        self.first_color = first_color
        self.second_color = second_color
        self.start_point = start_point
        self.end_point = end_point
        self.gradient_vector = end_point-start_point
        self.gradient_vector_length = norm(self.gradient_vector)
        tmp_array = []
    def project(self,y,x):
        return self.projection(array([y,x]),self.gradient_vector)/self.gradient_vector_length
    def projection(self,w,v):
        return (dot(v,w)/dot(v,v))*v
    def interpolate(self,t):
        return self.first_color + (self.second_color-self.first_color)*t
    def apply(self,img):
        image = deepcopy(img)

        if self.gradient_vector[0] >= 0:
            x_coordinates = array([np.linspace(0,1,img.shape[1]) for _ in range(img.shape[0])])
        else:
            x_coordinates = array([np.linspace(1,0,img.shape[1]) for _ in range(img.shape[0])])
        if self.gradient_vector[1] >= 0:
            y_coordinates = array([np.linspace(0,1,img.shape[0]) for _ in range(img.shape[1])]).transpose()
        else:
            y_coordinates = array([np.linspace(1,0,img.shape[0]) for _ in range(img.shape[1])]).transpose()
        print(f"Gradient vector: {self.gradient_vector}")

        # Calculate projections
        x_projections = abs(np.abs(x_coordinates * self.gradient_vector[0]))
        y_projections = abs(np.abs(y_coordinates * self.gradient_vector[1]))

        projection_matrix = x_projections + y_projections
        projection_matrix /= np.max(projection_matrix)

        for channel in range(3): # iterate through each color channel
            first_color = self.first_color[channel]
            second_color = self.second_color[channel]
            channel_gradient = first_color + ((second_color-first_color)*projection_matrix)
            channel_gradient /= np.max(channel_gradient)

            gradient_channel = image[:,:,channel] * channel_gradient
            image[:,:,channel] = gradient_channel.astype(int)
        return image

    def invapply(self,image):
        img = deepcopy(image)
        ĩmg = ~img
        inverted_first_color = array([1/self.first_color[0],1/self.first_color[1],1/self.first_color[2]])
        inverted_second_color = array([1/self.second_color[0],1/self.second_color[1],1/self.second_color[2]])
        inv_gradient = ColorGradient(inverted_first_color,
                                     inverted_second_color,
                                     self.start_point,
                                     self.end_point)
        return ~inv_gradient.apply(ĩmg)
    def alternate_inverted_apply(self,image): 
        img = deepcopy(image)
        ĩmg = ~img
        inverted_first_color = array([1/self.first_color[0],1/self.first_color[1],1/self.first_color[2]])
        inverted_second_color = array([1/self.second_color[0],1/self.second_color[1],1/self.second_color[2]])
        inv_gradient = ColorGradient(inverted_first_color,
                                     inverted_second_color,
                                     self.start_point,
                                     self.end_point)
        ĩmg = inv_gradient.apply(ĩmg)
        return ĩmg
    def weird_apply(self,im):
        img = deepcopy(im)
        i = self.apply(img)
        i = self.alternate_inverted_apply(i)
        return ~i
if __name__ == "__main__":
    def hex_to_rgb(s):
        # Converts a valid color code to a tuple of ints.
        # Throws ValueError if s is not a valid color code
        hexcode = s.lstrip('#')
        if len(hexcode) == 6:
            return array([int(hexcode[i:i+2], 16) for i in (0, 2, 4)])/255
        else:
            raise ValueError

    # Parse Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Name of image file to apply color gradient to")
    parser.add_argument("dest", help="Name of image file to write changes to")
    parser.add_argument("-c1", help="Hex Code of first color",default="#1111FF")
    parser.add_argument("-c2", help="Hex Code of second color", default="#11FF11")
    parser.add_argument("-a", "--angle", help="Angle gradient is applied at in degrees",default=0, type=float)
    parser.add_argument("-s", "--show", help="Display the new image. Image is still saved to file", action="store_true")
    application_mode = parser.add_mutually_exclusive_group()
    application_mode.add_argument("-n", "--normal", help="Apply gradient regularly", action="store_true")
    application_mode.add_argument("-d", "--dark", help="Apply gradient on darker colors rather than brighter ones.", action="store_true")
    application_mode.add_argument("-i", "---invert", help="Apply gradient to inverted image.", action="store_true")
    application_mode.add_argument("-w", "--weird", help="Do both the normal and inverted gradients.", action="store_true")


    args = parser.parse_args()

    filename = args.source
    destination_file = args.dest

    try:
        color1 = hex_to_rgb(args.c1)
    except:
        print(f"{args.c1} is not a valid color code. Examples of valid color codes: #c0ffee   badbad")
        exit(1)
    try:
        color2 = hex_to_rgb(args.c2)
    except:
        print(f"{args.c2} is not a valid color code. Examples of valid color codes: #c0ffee   badbad")
        exit(1)

    startpoint = array([0,0])
    endpoint = array([np.cos(np.radians(args.angle)),np.sin(np.radians(args.angle))])
    if args.angle < 0:
        startpoint[1] += 1
        endpoint[1] += 1

    # Load image
    img = cv.imread(filename)

    # Apply gradient to image
    if img is None:
        sys.exit("Could not read the image")
    color_gradient = ColorGradient(first_color = color1,
                                   second_color = color2,
                                   start_point = startpoint,
                                   end_point = endpoint)
    if args.normal:
        gradient_image = color_gradient.apply(img)
    elif args.dark:
        gradient_image = color_gradient.invapply(img)
    elif args.invert:
        gradient_image = color_gradient.alternate_inverted_apply(img)
    elif args.weird:
        gradient_image = color_gradient.weird_apply(img)
    else:
        gradient_image = color_gradient.apply(img)


    #invimg = ~color_gradient.invapply(img)
    cv.imwrite(destination_file,gradient_image)

    if args.show:
        cv.imshow("Display window",gradient_image)
        keypress = cv.waitKey(0)

        #if keypress == ord("s"):
        #    cv.imwrite("o"+filename, img)
