import cv2
import numpy as np
import imutils
import math
import matplotlib.pyplot as plt

            

def draw_contour(image, c):
    cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
   
        
    cv2.fillPoly(image, [c], (255, 0, 0))
    
    return image
    



def display_image(Image):
    Image_RGB = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    plt.imshow(Image_RGB)
    plt.show()



def preprocess(image):
    

    blur = cv2.GaussianBlur(image, (3,3), 1)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 72)
    display_image(canny)

    dilated = canny
    # kernel = np.ones((2))
    # dilated  = cv2.dilate(canny, kernel, iterations=1)

    # display_image(dilated)
    return dilated

# def find_contours(image_dilated):
#     thresh = cv2.threshold(image_dilated, 0, 255, cv2.THRESH_BINARY)[1]
#     contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     cnts = imutils.grab_contours(contours)

#     return cnts

def get_contours(image, image_contour):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fig = plt.figure(figsize=(12, 8))
    cnt = contours[0]

    for i in range(50):
        img_c = image_contour.copy()
        approx = cv2.approxPolyDP(cnt, i * 1, True)
        cv2.drawContours(img_c, [approx], -1, (255, 0, 255), 2)
        plt.imshow(img_c)
        plt.axis('off')
        plt.savefig(f"image_0{i}.png")

    return approx, image_contour, contours, cnt

def write_contour(contour, path):
    contour.tofile(path)


def find_shape(pic_contours, target_contour, peri_thres, target_img):

    min_value = 0.2
    for c in pic_contours:
        peri = cv2.arcLength(c, True)
        
        if peri <= peri_thres[0] or peri >= peri_thres[1]:
            continue
        Center = cv2.moments(c)
        
        if Center["m10"] == 0 or Center["m00"] == 0 or Center["m01"] == 0 or Center["m00"] == 0:
            continue
        cX = int((Center["m10"] / Center["m00"]) * 1)
        cY = int((Center["m01"] / Center["m00"]) * 1)
        
        
        ## Iterate through each contour in the target image and
        ###  use cv2.matchShape to compare contour shapes
        match = cv2.matchShapes(target_contour,c, 1, 0.0)
        print("peri", peri)
        ## if the match value is less than 0.15 we
        if match < min_value:
            
            min_value = match
            closest_contour = c
            target_img =  draw_contour(target_img, c)
            print("cX", cX)
            print("cY", cY) 
        
    return min_value, closest_contour

if __name__ == "__main__":
    triangle = cv2.imread('triangle.png', 0) 
    
    ### Load the traget image with the shapes we are trying to match
    image = cv2.imread('pic.png')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ### Threshold both images first before using cv2.findContours
    ret, thresh1 = cv2.threshold(triangle, 127, 255, 0)
    ret, thresh2 = cv2.threshold(image_gray, 100, 255, 0)

    # display_image(thresh1)
    # display_image(thresh2)
    ### Find contours in template
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


    ### We need to sort the contours by area so that we can remove the largest contour which is the image outline
    sorted_triangle_contours = sorted(contours, key = cv2.contourArea, reverse = True)

    ## We extract the second largest contour which will be our template contour (first largest will be the entire image)
    
    tri_contours = sorted_triangle_contours[1]

    
    ## Extract contours from full image
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    

    draw_contours = image.copy()

    min_value, closest_contour = find_shape(contours, tri_contours, (300, 900), draw_contours)
    print("min_value", min_value)
    
    draw_contour(image, closest_contour)
    # display_image(image)

    circle = cv2.imread("circle.png", 0)
    ret, thresh3 = cv2.threshold(circle, 100, 255, 0)
    # display_image(thresh3)
    
    blur = cv2.GaussianBlur(image_gray, (5, 5), 1)
    ret, thresh4 = cv2.threshold(blur, 60, 170, 0)
    display_image(thresh4)
    # canny = cv2.Canny(blur, 100, 170)
    # print("canny")
    # display_image(canny)
    
    canny_contours, hierarchy = cv2.findContours(thresh4, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


    circle_contours, hierarchy = cv2.findContours(thresh3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    

    ### We need to sort the contours by area so that we can remove the largest contour which is the image outline
    sorted_circle_contours = sorted(circle_contours, key = cv2.contourArea, reverse = True)

    ## We extract the second largest contour which will be our template contour (first largest will be the entire image)
    
    circle_contours = sorted_circle_contours[1]
    

    draw_circle = circle.copy()
    draw_contour(draw_circle, circle_contours)
    min_value, closest_contour = find_shape(canny_contours, circle_contours, (50, 200), draw_contours)
    print("min circle val", min_value)
    image = draw_contour(image, closest_contour)
    display_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imsave("./saved/by_shape2.png",image)
