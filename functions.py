import numpy as np


def decimal_to_binary(decimal_num):
    # Ensure the input is within the valid range
    if -256 <= decimal_num <= 255:
        # Handle negative values using two's complement
        if decimal_num < 0:
            binary_str = bin(2**9 + decimal_num)[2:]
        else:
            binary_str = bin(decimal_num)[2:]

        # Ensure the binary string is 11 bits long by adding leading zeros if needed
        binary_str = binary_str.zfill(9)

        return binary_str
    else:
        if decimal_num < 0:
            binary_str = bin(2**9 -256)[2:]
        else:
            binary_str = bin(255)[2:]

        binary_str = binary_str.zfill(9)

        return binary_str

        # raise ValueError("Input must be in the range -256 to 255 for an 9-bit binary number.",decimal_num )

def binary_to_decimal(binary_str):
    # Convert binary string to decimal
    decimal_num = int(binary_str, 2)
    
    # If the binary number is negative, convert it back using two's complement
    if binary_str[0] == '1':
        decimal_num -= 2**len(binary_str)
    
    return decimal_num


def run_length_coding(matrix):
    rle_list = []
    current_run = None
    run_length = 0

    for element in matrix:
        
            if current_run is None:
                current_run = element
                run_length = 1
            elif current_run == element:
                run_length += 1
            else:
                rle_list.append((decimal_to_binary(current_run), decimal_to_binary(run_length)))
                current_run = element
                run_length = 1

    # Add the last run
    rle_list.append((decimal_to_binary(current_run), decimal_to_binary(run_length)))
    # rle_list.append((current_run, run_length))
    return rle_list


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value


def count_characters_in_file(file_path):
    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the content of the file
            file_content = file.read()

            # Count the characters in the content
            character_count = len(file_content)

            return character_count

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def zigzag(matrix: np.ndarray) -> np.ndarray:
    # initializing the variables
    h = 0
    v = 0
    v_min = 0
    h_min = 0
    v_max = matrix.shape[0]
    h_max = matrix.shape[1]
    i = 0
    output = np.zeros((v_max * h_max))

    while (v < v_max) and (h < h_max):
        if ((h + v) % 2) == 0:  # going up
            if v == v_min:
                output[i] = matrix[v, h]  # first line
                if h == h_max:
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif (h == h_max - 1) and (v < v_max):  # last column
                output[i] = matrix[v, h]
                v = v + 1
                i = i + 1
            elif (v > v_min) and (h < h_max - 1):  # all other cases
                output[i] = matrix[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if (v == v_max - 1) and (h <= h_max - 1):  # last line
                output[i] = matrix[v, h]
                h = h + 1
                i = i + 1
            elif h == h_min:  # first column
                output[i] = matrix[v, h]
                if v == v_max - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif (v < v_max - 1) and (h > h_min):  # all other cases
                output[i] = matrix[v, h]
                v = v + 1
                h = h - 1
                i = i + 1
        if (v == v_max - 1) and (h == h_max - 1):  # bottom right element
            output[i] = matrix[v, h]
            break
    return output


def zigzag_decode(zigzag_array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    matrix = np.zeros((rows, cols))
    
    # initializing the variables
    h = 0
    v = 0
    v_min = 0
    h_min = 0
    v_max = rows
    h_max = cols
    i = 0

    while (v < v_max) and (h < h_max):
        if ((h + v) % 2) == 0:  # going up
            if v == v_min:
                matrix[v, h] = zigzag_array[i]  # first line
                if h == h_max:
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif (h == h_max - 1) and (v < v_max):  # last column
                matrix[v, h] = zigzag_array[i]
                v = v + 1
                i = i + 1
            elif (v > v_min) and (h < h_max - 1):  # all other cases
                matrix[v, h] = zigzag_array[i]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if (v == v_max - 1) and (h <= h_max - 1):  # last line
                matrix[v, h] = zigzag_array[i]
                h = h + 1
                i = i + 1
            elif h == h_min:  # first column
                matrix[v, h] = zigzag_array[i]
                if v == v_max - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif (v < v_max - 1) and (h > h_min):  # all other cases
                matrix[v, h] = zigzag_array[i]
                v = v + 1
                h = h - 1
                i = i + 1
        if (v == v_max - 1) and (h == h_max - 1):  # bottom right element
            matrix[v, h] = zigzag_array[i]
            break
    
    return matrix.astype(int)