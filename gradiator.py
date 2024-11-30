import cv2 as cv
import sys
import os
from numpy import dot, array, flip
import numpy as np
from numpy.linalg import norm
from copy import deepcopy

# Color gradient
class ColorGradient:
    first_color = array([0,0,0])
    second_color = array([1,1,1]) #[255,255,255]
    first_point = array([0,0])
    end_point = array([1,1])


    def __init__(self,first_color=array([0,0,0]),second_color=array([1,1,1]),first_point=array([0,0]),end_point=array([1,1])) -> None:
        self.first_color = first_color
        self.second_color = second_color
        self.first_point = first_point
        self.end_point = end_point
        self.gradient_vector = end_point-first_point
        self.gradient_vector_length = None
        tmp_array = []
        self.initiate_color_vector()
    def initiate_color_vector(self):
        self.gradient_vector = self.end_point-self.first_point
        tmp_array = []
        for i in [(0,0),(0,1),(1,0),(1,1)]:
            for j in [(0,0),(0,1),(1,0),(1,1)]:
                i2 = array([i[0],i[1]])
                j2 = array([j[0],j[1]])
                iv = self.projection(i2,self.gradient_vector)
                jv = self.projection(j2,self.gradient_vector)
                tmp_array.append(norm(iv-jv))
        self.gradient_vector_length = np.max(array(tmp_array))
    def project(self,y,x):
        return self.projection(array([y,x]),self.gradient_vector)/self.gradient_vector_length
    def projection(self,w,v):
        return (dot(v,w)/dot(v,v))*v
    def interpolate(self,t):
        return self.first_color + (self.second_color-self.first_color)*t
    def apply(self,img):
        image = deepcopy(img)

        x_coordinates = array([np.linspace(0,1,img.shape[1]) for _ in range(img.shape[0])])
        y_coordinates = array([np.linspace(0,1,img.shape[0]) for _ in range(img.shape[1])]).transpose()

        for channel in range(3): # We iterate through each color channel
            first_color = self.first_color[channel]
            second_color = self.second_color[channel]
            channel_gradient = first_color + (second_color-first_color)*x_coordinates
            print(channel_gradient)
            print(image[:,:,channel])

            gradient_channel = image[:,:,channel] * channel_gradient
            image[:,:,channel] = gradient_channel.astype(int)

        print(image - img)
        return image

        """
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                coordinate = array([y/img.shape[0],x/img.shape[1]]) # Normalize coordinate
                projection = norm(color_gradient.project(coordinate[0],coordinate[1]))
                #brightness = 1 - norm(projection)
                gradient = self.interpolate(projection)
                img[y,x] = img[y,x] * gradient
        """
    def invapply(self,img):
        ĩmg = ~img
        inverted_first_color = array([1/self.first_color[0],1/self.first_color[1],1/self.first_color[2]])
        inverted_second_color = array([1/self.second_color[0],1/self.second_color[1],1/self.second_color[2]])
        inv_gradient = ColorGradient(inverted_first_color,
                                     inverted_second_color,
                                     self.first_point,
                                     self.end_point)
        inv_gradient.apply(ĩmg)
        return ĩmg
    def alternate_inverted_apply(self,image): 
        img = deepcopy(image)
        ĩmg = ~img
        inverted_first_color = array([1/self.c1[0],1/self.c1[1],1/self.c1[2]])
        inverted_second_color = array([1/self.c2[0],1/self.c2[1],1/self.c2[2]])
        inv_gradient = ColorGradient(inverted_first_color,
                                     inverted_second_color,
                                     self.first_point,
                                     self.second_point)
        ĩmg = inv_gradient.apply(ĩmg)
        return ĩmg
    def doubleapp(self,im):
        img = deepcopy(im)
        i = self.apply(img)
        i = self.alternate_inverted_apply(i)
        return ~i
if __name__ == "__main__":
    # Parse Command Line Arguments:
    def help():
        print("Gradiator: The Command Line Color Gradient Applicator")
        print("Usage: ")
        sys.exit(0)
    filename = ""
    display_image = True
    output_directory = "output/"
    if len(sys.argv) >= 2:
        for arg in sys.argv[1:]:
            # Placeholder before actual CLI functionality is added
            if "." in arg:
                filename = arg
    else:
        help()
    if filename == "":
        help()
    # Last inn filen
    img = cv.imread(filename)
    # Vis bildet og avslutt programmet
    if img is None:
        sys.exit("Could not read the image")
    color_1 = flip(array([142,255,185])/255)
    color_2 = flip(array([255,226,128])/255)
    color_gradient = ColorGradient(first_color = color_1, second_color = color_2,end_point=array([1,0]))
    gradient_image = color_gradient.apply(img)
    #invimg = ~color_gradient.invapply(img)
    filename = filename.split('/')[-1]
    cv.imwrite(f"gradient-{filename}",gradient_image)

    if display_image:
        cv.imshow("Display window",gradient_image)
        keypress = cv.waitKey(0)

        #if keypress == ord("s"):
        #    cv.imwrite("o"+filename, img)
