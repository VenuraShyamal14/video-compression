import cv2
import numpy as np
import math
from scipy.fftpack import dct, idct

quantization_matrix_for_auto=[
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2]
]

# Specify the path to your image file
image_path = "original.jpg"

# Read the image using OpenCV in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
im_size = 560

def save_dict_to_txt(dictionary, file_path):
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

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

def count_bits(run_length_code, total_bits):
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

def img_comp(image, im_size, marcoBlockSize, quantized_matrix, total_bits):
    marcoBlock_dict = {}
    dct_dict = {}
    quantized_result = {}
    run_length_dict = {}
    inverse_dct_dict = {}
    dequantized_result = {}

    for i in range(math.ceil(im_size / marcoBlockSize)):
        for j in range(math.ceil(im_size / marcoBlockSize)):
            new_img = image[i * marcoBlockSize:i * marcoBlockSize + marcoBlockSize,
                      j * marcoBlockSize:j * marcoBlockSize + marcoBlockSize]
            marcoBlock_dict[str(i) + "_" + str(j)] = new_img.tolist()

            # Apply 2D DCT
            dct_transformed = dct(dct(new_img, axis=0, norm='ortho'), axis=1, norm='ortho')

            result_array = np.round(dct_transformed / quantized_matrix).astype(int)
            quantized_result[str(i) + "_" + str(j)] = result_array

            run_length_dict[str(i) + "_" + str(j)] = run_length_coding(result_array)
            total_bits = count_bits(run_length_dict[str(i) + "_" + str(j)], total_bits)

    save_dict_to_txt(run_length_dict, "encode.txt")

    decode_run_length_dict = read_and_decode_text_file("encode.txt")

    for key in decode_run_length_dict:
        decoded_matrix = decode_run_length_dict[key]
        result_array_1 = (np.array(decoded_matrix) * np.array(quantized_matrix))
        dequantized_result[key] = result_array_1
        inverse_dct_dict[key] = np.round(
            idct(idct(dequantized_result[key], axis=0, norm='ortho'), axis=1, norm='ortho')).astype(int)

    empty_image = np.zeros((im_size, im_size), dtype=np.uint8)

    for key in inverse_dct_dict:
        row = int(key.split("_")[0]) * marcoBlockSize
        colomn = int(key.split("_")[1]) * marcoBlockSize
        empty_image[row:row + marcoBlockSize, colomn:colomn + marcoBlockSize] = inverse_dct_dict[key]

    print("psnr value ", psnr(image, empty_image))
    return total_bits

total_bits = 0

def decrement_list_elements(matrix, value, x, y):
    matrix[x][y] -= value

def adjust_quantization_matrix(given_value, image, im_size, marco_block, quantization_matrix, total_bits):
    # Copy the original matrix to avoid modifying the input directly
    new_quantization_matrix = np.copy(quantization_matrix)

    while total_bits != given_value:
        # Iterate through each element in the matrix
        for i in range(new_quantization_matrix.shape[0]):
            for j in range(new_quantization_matrix.shape[1]):
                # Increment each element by 1
                new_quantization_matrix[i, j] += 1

                # Recalculate total_bits
                total_bits = img_comp(image, im_size, marco_block, new_quantization_matrix, 0)

                # Check if total_bits is now equal to given_value
                if total_bits == given_value:
                    return new_quantization_matrix, total_bits

                # If not equal, decrement the previous increment
                new_quantization_matrix[i, j] -= 1
    print("changed quantization_matrix", new_quantization_matrix)
    return new_quantization_matrix, total_bits

# Call the function
result_quantization_matrix, result_total_bits = adjust_quantization_matrix(
    300000, image, im_size, 8, quantization_matrix_for_auto, total_bits
)

print("Adjusted Quantization Matrix:")
print(result_quantization_matrix)
print("Total Bits after Adjustment:", result_total_bits)

given_value = 300000

# # Loop until the condition is satisfied
# if result_total_bits > given_value:
#     while not (given_value - 100 <= result_total_bits <= given_value + 100):
#         for i in range(8):
#             for j in range(8):
#                 decrement_list_elements(quantization_matrix_for_auto, 1, i, j)
#                 result_total_bits = img_comp(image, im_size, 8, quantization_matrix_for_auto, result_total_bits)
#                 print(result_total_bits)
#                 if given_value - 100 <= result_total_bits <= given_value + 100:
#                     break
# else:
#     while not (given_value - 100 <= result_total_bits <= given_value + 100):
#         for i in range(8):
#             for j in range(8):
#                 decrement_list_elements(quantization_matrix_for_auto, -1, i, j)
#                 result_total_bits = img_comp(image, im_size, 8, quantization_matrix_for_auto, result_total_bits)
#                 print(result_total_bits)
#                 if given_value - 100 <= result_total_bits <= given_value + 100:
#                     break

# Print the final quantization matrix value
print("Final Quantization Matrix:")
print(quantization_matrix_for_auto)
