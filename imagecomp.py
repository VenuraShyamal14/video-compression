import cv2
import numpy as np
import math

# Specify the path to your image file
image_path = "black_and_white.jpg"


# Read the image using OpenCV in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
im_size=560
start=0
image = image[start:start+im_size,start:start+im_size]

def saveToText(name,img):
    output_file_path = name+".txt"
    np.savetxt(output_file_path, img, fmt='%d', delimiter=' ')

saveToText("original",image)

marcoBlock_dict={}

# Check if the image was successfully loaded
if image is not None:
    # Display the image
    # cv2.imshow("Image", image)
    # print(image[:,:])
    
    marcoBlockSize=280

    # print((math.ceil(im_size/marcoBlockSize)))
    for i in range(math.ceil(im_size/marcoBlockSize)):
        for j in range(math.ceil(im_size/marcoBlockSize)):
            new_img=image[i*marcoBlockSize:i*marcoBlockSize+marcoBlockSize,j*marcoBlockSize:j*marcoBlockSize+marcoBlockSize]
            # saveToText(str(i)+"_"+str(j),new_img)
            marcoBlock_dict[str(i)+"_"+str(j)]=new_img.tolist()

    # print(marcoBlock_dict["0_0"][279])

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Error: Unable to load the image from {image_path}")
