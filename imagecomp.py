import cv2
import numpy as np
import math
from scipy.fftpack import dct
from scipy.fftpack import idct

from functions import *

image_path = "original.jpg"

# Read the image using OpenCV in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
im_size=560
marcoBlockSize=8

required_bit_rate=(300+34)*1000  #bits    e18034

# saveToText("original",image)

marcoBlock_dict={}
dct_dict={}
ac_values_dict={}
# quantized_result={}
run_length_dict={}
inverse_dct_dict={}
dequantized_result={}

################################################# answer to 3.1.1  #############################

high_quaity_quantized_matrix=[
    [7, 4, 4, 4, 5, 8, 10, 12],
    [4, 4, 4, 4, 5, 12, 12, 11],
    [4, 4, 4, 5, 8, 11, 14, 11],
    [4, 4, 4, 6, 10, 17, 16, 12],
    [4, 4, 7, 11, 14, 22, 21, 15],
    [5, 7, 11, 13, 16, 12, 23, 18],
    [10, 13, 16, 17, 21, 24, 24, 21],
    [14, 18, 19, 20, 22, 20, 20, 20]
]

medium_quaity_quantized_matrix=[
    [8, 5, 5, 8, 12, 20, 26, 31],
    [6, 7, 7, 10, 13, 29, 30, 27],
    [7, 7, 8, 12, 20, 29, 35, 28],
    [7, 9, 11, 15, 26, 44, 41, 32],
    [9, 11, 19, 29, 35, 55, 52, 39],
    [12, 17, 26, 30, 38, 49, 56, 46],
    [24, 32, 39, 43, 50, 58, 58, 50],
    [36, 46, 48, 50, 57, 50, 52, 50]
]

low_quaity_quantized_matrix=[
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]

quantization_matrix_for_given_bitrate=[
    [25, 30, 35, 30, 44, 40, 51, 61],
    [29, 31, 35, 30, 46, 58, 60, 89],
    [35, 35, 39, 50, 60, 77, 69, 77],
    [50, 60, 64, 69, 81, 97, 80, 99],
    [68, 72, 77, 86, 98, 109, 173, 177],
    [74, 75, 85, 89, 91, 104, 113, 122],
    [79, 84, 88, 97, 103, 121, 120, 121],
    [92, 99, 99, 98, 112, 100, 103, 129]
]


########################################################### answer to 3.1.2 ####################
quantization_matrix_for_auto = [
    [25, 18, 10, 16, 24, 40, 51, 61],
    [18, 18, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]


# Check if the image was successfully loaded


def img_comp(image,im_size,quantized_matrix):
    with open('encode.txt', 'w') as file:
        file.write("")

    if image is not None:
        for i in range(math.ceil(im_size/marcoBlockSize)):
            for j in range(math.ceil(im_size/marcoBlockSize)):
                new_img=image[i*marcoBlockSize:i*marcoBlockSize+marcoBlockSize,j*marcoBlockSize:j*marcoBlockSize+marcoBlockSize]
                marcoBlock_dict[str(i)+"_"+str(j)]=new_img.tolist()
                # Apply 2D DCT
                dct_transformed = dct(dct(new_img, axis=0, norm='ortho'), axis=1, norm='ortho')
                
                # saveToText("dct_"+str(i)+"_"+str(j),dct_transformed)
                # saveToText(str(i)+"_"+str(j),new_img)
                dct_dict[str(i)+"_"+str(j)]=dct_transformed.tolist()
                # save_dict_to_txt(dct_dict,"dct_"+str(i)+"_"+str(j)+".txt")

                result_array = np.round(dct_transformed / quantized_matrix).astype(int)
                # quantized_result[str(i)+"_"+str(j)]=result_array[1:]
                # ac_values_dict[str(i)+"_"+str(j)]=result_array[0][0]
                output=zigzag(result_array).astype(int)
                # print(output)
                # if i==10 and j==0:
                #     print(result_array)

                # run_length_dict[str(i)+"_"+str(j)]=run_length_coding(result_array[1:])
                run_length_dict[str(i)+"_"+str(j)]=run_length_coding(output)
                with open('encode.txt', 'a') as file:
                    for i1 in run_length_dict[str(i)+"_"+str(j)]:
                        file.write(f"{str(i1[0])+str(i1[1])}")
        
        
        
        
        
        # print(run_length_dict["0_43"])
        # print("marco_block",marcoBlock_dict["0_0"])
        # print("dct",dct_dict["0_0"])
        # print("decoded_quantised_result",quantized_result["0_0"])
        # print("run length",run_length_dict["0_0"])
        
        
    else:
        print(f"Error: Unable to load the image from {image_path}")

def read_and_decode_text_file(file_path):
    decode_dict = {}
    decode_run_length_list=[]

    with open(file_path, 'r') as file:
        for line in file:
            value_str = line.strip()
            print("# characters in the file",len(value_str))
        
            while len(value_str) !=0:
                run=value_str[0:9]
                length=value_str[9:18]
                value_str=value_str[18:]
                for i2 in range(binary_to_decimal(length)):
                    decode_run_length_list.append(binary_to_decimal(run))
                # total=binary_to_decimal(length)+total
    
            for i3 in range(math.ceil(im_size/marcoBlockSize)):
                for j3 in range(math.ceil(im_size/marcoBlockSize)):
                    decode_macro_block=decode_run_length_list[0:64]
                    inverse_zig_zag=zigzag_decode(decode_macro_block,8,8)
                    decode_dict[str(i3)+"_"+str(j3)]=inverse_zig_zag
                    decode_run_length_list=decode_run_length_list[64:]
            

            

            

            # Decode the run-length code to get the 8x8 matrix
            # decoded_matrix = run_length_decode(rle_list)

            # Reshape the 1D array to a 2D 8x8 matrix
            # decoded_matrix = [decoded_matrix[i:i + 8] for i in range(0, len(decoded_matrix), 8)]

            # Store the key and decoded matrix in the dictionary
            # decode_dict[key] = decoded_matrix

    # print("decode_run_length_list",decode_run_length_list)
    # print("total",len(decode_run_length_list))
    return decode_dict



def create_img(decode_run_length_dict,quantized_matrix):
        empty_image = np.zeros((im_size, im_size), dtype=np.uint8)

        for key in decode_run_length_dict:
            decoded_matrix=decode_run_length_dict[key]
            result_array_1 = (np.array(decoded_matrix) * np.array(quantized_matrix))
            dequantized_result[key]=result_array_1
            inverse_dct_dict[key]= np.round(idct(idct(dequantized_result[key], axis=0, norm='ortho'), axis=1, norm='ortho')).astype(int)
        

        
        # print("length of ", str(len(inverse_dct_dict)))
        for key in inverse_dct_dict:
            row=int(key.split("_")[0])*marcoBlockSize
            colomn=int(key.split("_")[1])*marcoBlockSize
            empty_image[row:row+marcoBlockSize,colomn:colomn+marcoBlockSize]=inverse_dct_dict[key]

        # saveToText("after",empty_image)
        
        # image1 = cv2.imread(empty_image)
        cv2.imshow("decoded image", empty_image)

        # print("dequantised_result",dequantized_result["0_0"])
        # print("inverse_dct",inverse_dct_dict["0_0"])

        # Save the image using OpenCV
        cv2.imwrite("test.jpg", empty_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return empty_image

        # # Wait for a key press and then close the window
        

img_comp(image,im_size,high_quaity_quantized_matrix)
d=read_and_decode_text_file("encode.txt")
c=create_img(d,high_quaity_quantized_matrix)
# print(d["10_0"])
################################################ answer to  3.1.2 ###############################
print("total bits in encoded file ",count_characters_in_file("encode.txt"))
print("psnr value ",psnr(image, c))

################################################ answer to  3.1.3 ###############################

given_value = 334000

# Loop until the condition is satisfied
# while not (given_value - 10000 <= t <= given_value + 10000):
#     if t < given_value:
#         quantization_matrix_for_auto = [[element - 1 for element in row] for row in quantization_matrix_for_auto]
#         for i in range(len(quantization_matrix_for_auto)):
#             for j in range(len(quantization_matrix_for_auto[0])):
#                 if quantization_matrix_for_auto[i][j] == 0:
#                     quantization_matrix_for_auto[i][j] = 1

#         # print(quantization_matrix_for_auto)
#         total_bits=0
#         t=img_comp(image,im_size,8,quantization_matrix_for_auto,0)
#     else:
#         quantization_matrix_for_auto = [[element + 1 for element in row] for row in quantization_matrix_for_auto]
#         for i in range(len(quantization_matrix_for_auto)):
#             for j in range(len(quantization_matrix_for_auto[0])):
#                 if quantization_matrix_for_auto[i][j] == 0:
#                     quantization_matrix_for_auto[i][j] = 1
#         # print(quantization_matrix_for_auto)
#         total_bits=0
#         t=img_comp(image,im_size,8,quantization_matrix_for_auto,0)

#     print(quantization_matrix_for_auto)
#     print(t)

# print("quantization matrix for given",given_value,"bitrate ",quantization_matrix_for_auto)
# print("final bits for ",given_value,t)