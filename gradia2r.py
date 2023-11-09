import cv2 as cv
import sys
import os
from numpy import dot, array, flip, max as mx
from numpy.linalg import norm

print("the gradiator has started")
# Last inn filen
filnavn = "berserk1.jpg"
os.chdir("/home/ascend/Documents/personelig/gradiator")
relative_path = os.path.join("images",filnavn)
absolute_path = os.path.join(os.getcwd(),relative_path)
img = cv.imread((absolute_path))
# Vis bildet og avslutt programmet
if img is None:
    sys.exit("Could not read the image")
# Color gradient
class ColorGradient:
    c1 = array([0,0,0])
    c2 = array([1,1,1]) #[255,255,255]
    p1 = array([0,0])
    p2 = array([1,1])


    def __init__(self,c1=array([0,0,0]),c2=array([1,1,1]),p1=array([0,0]),p2=array([1,1])) -> None:
        self.c1 = c1
        self.c2 = c2
        self.p1 = p1
        self.p2 = p2
        self.v = p2-p1
        tmp_array = []
        for i in [(0,0),(0,1),(1,0),(1,1)]:
            for j in [(0,0),(0,1),(1,0),(1,1)]:
                i2 = array([i[0],i[1]])
                j2 = array([j[0],j[1]])
                iv = self.projection(i2,self.v)
                jv = self.projection(j2,self.v)
                tmp_array.append(norm(iv-jv))
        self.v_length = mx(array(tmp_array))
    def update(self):
        self.v = self.p2-self.p1
        tmp_array = []
        for i in [(0,0),(0,1),(1,0),(1,1)]:
            for j in [(0,0),(0,1),(1,0),(1,1)]:
                i2 = array([i[0],i[1]])
                j2 = array([j[0],j[1]])
                iv = self.projection(i2,self.v)
                jv = self.projection(j2,self.v)
                tmp_array.append(norm(iv-jv))
        self.v_length = mx(array(tmp_array))
    def project(self,y,x):
        return self.projection(array([y,x]),self.v)/self.v_length#/max_self.projection #+self.projection(array([0,0]),self.p2-self.p1)
    def projection(self,w,v):
        return (dot(v,w)/dot(v,v))*v
    def interpolate(self,t):
        return self.c1 + (self.c2-self.c1)*t
    def apply(self,img):
        self.update()
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                coordinate = array([y/img.shape[0],x/img.shape[1]])
                projection = norm(color_gradient.project(coordinate[0],coordinate[1]))
                #brightness = 1 - norm(projection)
                gradient = self.interpolate(projection)
                img[y,x] = img[y,x] * gradient
    def invapply(self,img):
        ĩmg = ~img
        invc1 = array([1/self.c1[0],1/self.c1[1],1/self.c1[2]])
        invc2 = array([1/self.c2[0],1/self.c2[1],1/self.c2[2]])
        inv_gradient = ColorGradient(invc1,invc2,self.p1,self.p2)
        inv_gradient.apply(ĩmg)
        return ĩmg

color_1 = flip(array([142,255,185])/255)
color_2 = flip(array([255,226,128])/255)
color_gradient = ColorGradient(c1 = color_1, c2 = color_2,p2=array([1,0]))
gradient = img
color_gradient.apply(gradient)
invimg = ~color_gradient.invapply(img)
cv.imwrite(f"{filnavn}",gradient)
cv.imwrite(f"inverted {filnavn}",invimg)

cv.imshow("Display window",img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("o"+filnavn, img)
