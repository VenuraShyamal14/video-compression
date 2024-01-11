import cv2
import numpy as np
import math
from scipy.fftpack import dct
from scipy.fftpack import idct


# Specify the path to your image file
# image_path = "black_and_white.jpg"
image_path = "original.jpg"

# Read the image using OpenCV in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
im_size=560

required_bit_rate=(300+34)*1000  #bits    e18034
# start=0
# image = image[start:start+im_size,start:start+im_size]
# Save the image using OpenCV
# cv2.imwrite("original.jpg", image)

def save_dict_to_txt(dictionary, file_path):
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

def saveToText(name,img):
    output_file_path = name+".txt"
    np.savetxt(output_file_path, img, fmt='%d', delimiter=' ')

def run_length_coding(matrix):
    rle_list = []
    current_run = None
    run_length = 0

    for row in matrix:
        for element in row:
            if current_run is None:
                current_run = element
                run_length = 1
            elif current_run == element:
                run_length += 1
            else:
                rle_list.append((current_run, run_length))
                current_run = element
                run_length = 1

    # Add the last run
    rle_list.append((current_run, run_length))
    return rle_list

def run_length_decode(rle_list):
    decoded_matrix = []

    for element, run_length in rle_list:
        decoded_matrix.extend([element] * run_length)

    return decoded_matrix

def read_and_decode_text_file(file_path):
    decode_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by ':' to get the key and value part
            key, value_str = line.strip().split(':')
            key = key.strip()

            # Extract the values part and remove leading/trailing characters
            value_str = value_str.strip()[1:-1]

            # Handle the case where there's no comma after the opening parenthesis
            value_str = value_str.replace('),', ')|').replace('(', '').replace(')', '')

            # Split the string into pairs and convert to a list of tuples
            rle_list = [tuple(map(int, pair.split(','))) for pair in value_str.split('|')]

            # Decode the run-length code to get the 8x8 matrix
            decoded_matrix = run_length_decode(rle_list)

            # Reshape the 1D array to a 2D 8x8 matrix
            decoded_matrix = [decoded_matrix[i:i + 8] for i in range(0, len(decoded_matrix), 8)]

            # Store the key and decoded matrix in the dictionary
            decode_dict[key] = decoded_matrix

    return decode_dict

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

total_bits=0
def count_bits(run_length_code,total_bits):
    
    for run in run_length_code:
        # Each run is represented by a tuple (value, length)
        value, length = run
        # Calculate the number of bits required to represent the value
        value_bits = len(bin(value)[2:])
        # Calculate the number of bits required to represent the length
        length_bits = len(bin(length)[2:])
        # Add the bits required for both value and length
        total_bits += value_bits + length_bits
    return total_bits


# saveToText("original",image)

marcoBlock_dict={}
dct_dict={}
quantized_result={}
run_length_dict={}
inverse_dct_dict={}
dequantized_result={}



high_quaity_quantized_matrix=[
    [3, 2, 2, 3, 5, 8, 10, 12],
    [2, 2, 3, 4, 5, 12, 12, 11],
    [3, 3, 3, 5, 8, 11, 14, 11],
    [3, 3, 4, 6, 10, 17, 16, 12],
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
    [1, 1, 1, 1, 1, 1, 2, 2],
    [1, 1, 1, 1, 1, 1, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2]
]

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
empty_image = np.zeros((im_size, im_size), dtype=np.uint8)
def img_comp(image,im_size,marcoBlockSize,quantized_matrix,total_bits):
    if image is not None:
        # Display the image
        # cv2.imshow("Image", image)
        # print(image[:,:])
        
        # marcoBlockSize=8
        # quantized_matrix=quantization_matrix_for_given_bitrate

        # print((math.ceil(im_size/marcoBlockSize)))
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
                quantized_result[str(i)+"_"+str(j)]=result_array

                run_length_dict[str(i)+"_"+str(j)]=run_length_coding(result_array)
                total_bits=count_bits(run_length_dict[str(i)+"_"+str(j)],total_bits)
                
                # print(result_list)

        # print("total bits after encode",total_bits)
        
        # save_dict_to_txt(run_length_dict,"encode.txt")
        # print(dct_dict["0_0"])
        # print("marco_block",marcoBlock_dict["0_0"])
        # print("dct",dct_dict["0_0"])
        # print("decoded_quantised_result",quantized_result["0_0"])
        # print(run_length_dict["0_0"])
        
        decode_run_length_dict = read_and_decode_text_file("encode.txt")
        # print("run_length_code",decode_run_length_dict["0_0"])


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
        # cv2.imshow("Image", empty_image)

        # print("dequantised_result",dequantized_result["0_0"])
        # print("inverse_dct",inverse_dct_dict["0_0"])

        # Save the image using OpenCV
        # cv2.imwrite("test.jpg", empty_image)

        
        return total_bits

        # # Wait for a key press and then close the window
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print(f"Error: Unable to load the image from {image_path}")


t=img_comp(image,im_size,8,quantization_matrix_for_given_bitrate,total_bits)
print("total bits after encode",t)
print("psnr value ",psnr(image, empty_image))


given_value = 400000

# Loop until the condition is satisfied
while not (given_value - 10000 <= t <= given_value + 10000):
    if t < given_value:
        quantization_matrix_for_auto = [[element - 1 for element in row] for row in quantization_matrix_for_auto]
        for i in range(len(quantization_matrix_for_auto)):
            for j in range(len(quantization_matrix_for_auto[0])):
                if quantization_matrix_for_auto[i][j] == 0:
                    quantization_matrix_for_auto[i][j] = 1

        # print(quantization_matrix_for_auto)
        total_bits=0
        t=img_comp(image,im_size,8,quantization_matrix_for_auto,0)
    else:
        quantization_matrix_for_auto = [[element + 1 for element in row] for row in quantization_matrix_for_auto]
        for i in range(len(quantization_matrix_for_auto)):
            for j in range(len(quantization_matrix_for_auto[0])):
                if quantization_matrix_for_auto[i][j] == 0:
                    quantization_matrix_for_auto[i][j] = 1
        # print(quantization_matrix_for_auto)
        total_bits=0
        t=img_comp(image,im_size,8,quantization_matrix_for_auto,0)
        
    print(quantization_matrix_for_auto)
    print(t)

print("quantization matrix for given",given_value,"bitrate ",quantization_matrix_for_auto)
print("final bits for ",given_value,t)