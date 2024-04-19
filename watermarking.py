import os
import xlwt
import shutil
import cv2
import sys
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import signal
import random
import base64

quant = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

class RPEEncryptDecrypt:
    @staticmethod
    def text_to_binary(message):
        return ''.join(format(ord(char), '08b') for char in message)

    @staticmethod
    def binary_to_text(binary_str):
        return ''.join([chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8)])

    @staticmethod
    def encode_rpe(original_image, secret_message):
        binary_message = RPEEncryptDecrypt.text_to_binary(secret_message)

        if len(binary_message) > original_image.size[0] * original_image.size[1] * 3:
            raise ValueError("Message is too long to be encoded in the image.")

        pixels = list(original_image.getdata())
        random.seed(42)  # Setting a seed for consistency

        for i in range(len(binary_message)):
            pixel_index = random.randint(0, len(pixels) - 1)
            pixel_value = list(pixels[pixel_index])
            channel_index = random.randint(0, 2)  # Randomly select a channel (R, G, or B)
            pixel_value[channel_index] &= 254  # Clear the least significant bit
            pixel_value[channel_index] |= int(binary_message[i])  # Set the least significant bit
            pixels[pixel_index] = tuple(pixel_value)

        new_img = Image.new(original_image.mode, original_image.size)
        new_img.putdata(pixels)
        return new_img

    @staticmethod
    def decode_rpe(encoded_image):
        pixels = list(encoded_image.getdata())

        binary_message = ''
        for pixel in pixels:
            for value in pixel:
                binary_message += str(value & 1)  # Extracting the least significant bit

        message = RPEEncryptDecrypt.binary_to_text(binary_message)
        return message

class DWT():
    def encode_image(self, img, secret_msg, length):
        bImg = self.iwt2(img)
        height, width = bImg.shape[:2]
        index = 0
        for row in range(height):
            for col in range(width):
                if index < len(secret_msg):
                    c = secret_msg[index]
                    asc = ord(c)
                else:
                    asc = bImg[row, col]
                bImg[row, col] = asc
                index += 1
        return bImg

    def decode_image(self, img, length):
        # Decoding the hidden message from the DWT encoded image
        msg = ""
        # Get size of image in pixels
        height, width = img.shape[:2]
        index = 0
        for row in range(height):
            for col in range(width):
                if index <= length:
                    # Assuming the message is encoded in the blue channel
                    pixel_value = img[row, col, 0]
                    msg += chr(pixel_value)
                    index += 1
        return msg


    def _iwt(self, array):
        output = np.zeros_like(array)
        nx, ny = array.shape[:2]
        x = nx // 2
        for j in range(ny):
            output[0:x, j] = (array[0:x*2:2, j] + array[1:x*2:2, j]) // 2
            output[x:nx, j] = array[0:x*2:2, j] - array[1:x*2:2, j]
        return output


    def _iiwt(self, array):
        output = np.zeros_like(array)
        nx, ny = array.shape
        x = nx // 2
        for j in range(ny):
            output[0::2, j] = array[0:x, j] + (array[x:nx, j] + 1) // 2
            output[1::2, j] = output[0::2, j] - array[x:nx, j]
        return output

    def iwt2(self, array):
        array = array.squeeze()  # Remove single-dimensional entries from the shape of an array
        return self._iwt(self._iwt(array.astype(int)).T).T


    def iiwt2(self, array):
        return self._iiwt(self._iiwt(array.astype(int).T).T)


class DCT():
    def __init__(self):
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0

    def encode_image(self, img, secret_msg):
        row, col = img.shape[:2]
        self.message = str(len(secret_msg)) + '*' + secret_msg
        self.bitMess = self.toBits()
        self.oriRow, self.oriCol = row, col
        if ((col / 8) * (row / 8) < len(secret_msg)):
            print("Error: Message too large to encode in image")
            return False
        if row % 8 != 0 or col % 8 != 0:
            img = self.addPadd(img, row, col)
        row, col = img.shape[:2]
        bImg, gImg, rImg = cv2.split(img)
        bImg = np.float32(bImg)
        imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j, i) in itertools.product(range(0, row, 8),
                                                                       range(0, col, 8))]
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        quantizedDCT = [np.round(dct_Block / quant) for dct_Block in dctBlocks]
        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            DC[7] = self.bitMess[messIndex][letterIndex]
            DC = np.packbits(DC)
            DC = np.float32(DC)
            DC = DC - 255
            quantizedBlock[0][0] = DC
            letterIndex = letterIndex + 1
            if letterIndex == 8:
                letterIndex = 0
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break
        sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        return sImg

    def decode_image(self, img):
        row, col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0
        bImg, gImg, rImg = cv2.split(img)
        bImg = np.float32(bImg)
        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j, i) in itertools.product(range(0, row, 8),
                                                                              range(0, col, 8))]
        quantizedDCT = [img_Block / quant for img_Block in imgBlocks]
        i = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            if DC[7] == 1:
                buff += (0 & 1) << (7 - i)
            elif DC[7] == 0:
                buff += (1 & 1) << (7 - i)
            i = 1 + i
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i = 0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                    except:
                        pass
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize)) + 1:]
        sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        return ''

    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

    def addPadd(self, img, row, col):
        img = cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))
        return img

    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8, '0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8, '0')
        return bits


class LSB():
    #encoding part :
    def encode_image(self,img, msg):
        length = len(msg)
        if length > 255:
            print("text too long! (don't exeed 255 characters)")
            return False
        encoded = img.copy()
        width, height = img.size
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode != 'RGB':
                    r, g, b ,a = img.getpixel((col, row))
                elif img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))
                # first value is length of msg
                if row == 0 and col == 0 and index < length:
                    asc = length
                elif index <= length:
                    c = msg[index -1]
                    asc = ord(c)
                else:
                    asc = b
                encoded.putpixel((col, row), (r, g , asc))
                index += 1
        return encoded
    
    #decoding part :
    def decode_image(self,img):
        width, height = img.size
        msg = ""
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode != 'RGB':
                    r, g, b ,a = img.getpixel((col, row))
                elif img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))  
                # first pixel r value is length of message
                if row == 0 and col == 0:
                    length = b
                elif index <= length:
                    msg += chr(b)
                index += 1
        lsb_decoded_image_file = "lsb_" + original_image_file
        #img.save(lsb_decoded_image_file)
        ##print("Decoded image was saved!")
        return msg

class SpreadSpectrumSteganography:
    def __init__(self, strength=1):
        self.strength = strength
        self.pseudo_random_seq = None

    def encrypt(self, cover_image, secret_message):
        # Convert secret message to binary
        secret_message_binary = ''.join(format(ord(char), '08b') for char in secret_message)
        
        # Get the dimensions of the cover image
        height, width = cover_image.size
        
        # Calculate the number of bits we can encode
        max_bits_to_encode = height * width * 3 * self.strength
        
        if len(secret_message_binary) > max_bits_to_encode:
            raise ValueError("Message too large to be encoded in the given image with the specified strength")
        
        # Convert the cover image to numpy array
        cover_image_array = np.array(cover_image)
        
        # Generate pseudo-random sequence based on image shape
        self.generate_pseudo_random_sequence(cover_image_array.shape[:2])

        # Spread spectrum encryption
        encrypted_image = cover_image_array.copy()
        idx = 0
        for row in range(encrypted_image.shape[0]):
            for col in range(encrypted_image.shape[1]):
                if idx < len(secret_message_binary):
                    # Apply spread spectrum encoding
                    pixel_val = list(encrypted_image[row, col])
                    pixel_val[0] ^= int(secret_message_binary[idx]) ^ self.pseudo_random_seq[row, col]
                    encrypted_image[row, col] = tuple(pixel_val)
                    idx += 1
                else:
                    break
            else:
                continue
            break
        
        return encrypted_image

    def decrypt(self, encrypted_image):
        # Decode the spread spectrum encoded message from the image
        binary_msg = ''
        for row in range(encrypted_image.shape[0]):
            for col in range(encrypted_image.shape[1]):
                pixel_val = np.all(encrypted_image[row, col][0])  # Accessing the first element of the pixel value
                pseudo_random_val = np.any(self.pseudo_random_seq[row, col])  # Accessing the corresponding pseudo-random sequence value
                binary_msg += '1' if (pixel_val ^ pseudo_random_val) else '0'

        # Pad the binary message to ensure it's a multiple of 8
        remainder = len(binary_msg) % 8
        if remainder != 0:
            binary_msg = binary_msg[:-(remainder)]  # Remove excess bits

        # Decode the binary message to ASCII characters
        decoded_msg = ''
        for i in range(0, len(binary_msg), 8):
            decoded_msg += chr(int(binary_msg[i:i + 8], 2))
        
        return decoded_msg

    def generate_pseudo_random_sequence(self, img_shape):
        # Generate a pseudo-random sequence based on the image shape
        # You can use any suitable method to generate the sequence
        # For simplicity, let's use a deterministic sequence for demonstration
        np.random.seed(123)  # Seed for reproducibility
        self.pseudo_random_seq = np.random.randint(0, 2, size=img_shape)

class Compare():
    def correlation(self, img1, img2):
        if len(img1.shape) > 2:
            img1 = np.mean(img1, axis=2)
        if len(img2.shape) > 2:
            img2 = np.mean(img2, axis=2)
        
        return signal.correlate2d(img1, img2, mode='valid')

    def meanSquareError(self, img1, img2):
        error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        error /= float(img1.shape[0] * img1.shape[1])
        return error

    def psnr(self, img1, img2):
        mse = self.meanSquareError(img1, img2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    def embedding_capacity(self, img_size):
        # Number of bits that can be embedded per pixel
        return img_size * 3


if __name__ == "__main__":
    if os.path.exists("Encoded_image/"):
        shutil.rmtree("Encoded_image/")
    if os.path.exists("Decoded_output/"):
        shutil.rmtree("Decoded_output/")
    if os.path.exists("Comparison_result/"):
        shutil.rmtree("Comparison_result/")
    os.makedirs("Encoded_image/")
    os.makedirs("Decoded_output/")
    os.makedirs("Comparison_result/")

    original_image_file = ""
    lsb_encoded_image_file = ""
    dct_encoded_image_file = ""
    dwt_encoded_image_file = ""
    spread_spectrum_encoded_image_file = ""
    rpe_encoded_image_file = ""  # Corrected initialization

    while True:
        m = input("To encode press '1', to decode press '2', to compare press '3', press any other button to close: ")

        if m == "1":
            os.chdir("Original_image/")
            original_image_file = input("Enter the name of the file with extension : ")
            img = Image.open(original_image_file)
            lsb_img = Image.open(original_image_file)
            dct_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
            dwt_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
            spread_spectrum_img = Image.open(original_image_file)
            print("Description : ", lsb_img, "\nMode : ", lsb_img.mode)
            secret_msg = input("Enter the message you want to hide: ")
            print("The message length is: ", len(secret_msg))
            os.chdir("..")
            os.chdir("Encoded_image/")
            # Encoding using different methods
            lsb_img_encoded = LSB().encode_image(lsb_img, secret_msg)
            dct_img_encoded = DCT().encode_image(dct_img, secret_msg)
            dwt_img_encoded = DWT().encode_image(dwt_img, secret_msg, len(secret_msg))
            # Integration of Spread Spectrum
            spread_spectrum_steganography = SpreadSpectrumSteganography()
            encrypted_image = spread_spectrum_steganography.encrypt(spread_spectrum_img, secret_msg)
            rpe_img_encoded_array = RPEEncryptDecrypt().encode_rpe(img, secret_msg)


            lsb_encoded_image_file = "lsb_" + original_image_file
            lsb_img_encoded.save(lsb_encoded_image_file)
            dct_encoded_image_file = "dct_" + original_image_file
            cv2.imwrite(dct_encoded_image_file, dct_img_encoded)
            dwt_encoded_image_file = "dwt_" + original_image_file
            cv2.imwrite(dwt_encoded_image_file, dwt_img_encoded)
            spread_spectrum_encoded_image_file = "spread_spectrum_" + original_image_file
            Image.fromarray(encrypted_image).save(spread_spectrum_encoded_image_file)
            rpe_encoded_image_file = "rpe_" + original_image_file
            Image.fromarray(np.uint8(rpe_img_encoded_array)).save(rpe_encoded_image_file)

            print("Encoded images were saved!")
            os.chdir("..")

        # Decoding Section
        elif m == "2":
            os.chdir("Encoded_image/")
            # Load encoded images
            lsb_img = Image.open(lsb_encoded_image_file)
            dct_img = cv2.imread(dct_encoded_image_file, cv2.IMREAD_UNCHANGED)
            dwt_img = cv2.imread(dwt_encoded_image_file, cv2.IMREAD_UNCHANGED)
            spread_spectrum_img = Image.open(spread_spectrum_encoded_image_file)
            rpe_img = Image.open(rpe_encoded_image_file)
            os.chdir("..")
            os.makedirs("Decoded_output/", exist_ok=True)
            os.chdir("Decoded_output/")

            # Decoding using different methods
            lsb_hidden_text = LSB().decode_image(lsb_img)
            dct_hidden_text = DCT().decode_image(dct_img)
            dwt_hidden_text = DWT().decode_image(dwt_img, len(secret_msg))
            spread_spectrum_steganography = SpreadSpectrumSteganography()
            spread_spectrum_steganography.generate_pseudo_random_sequence(np.array(encrypted_image).shape)
            encrypted_image_array = np.array(spread_spectrum_img)
            decrypted_message = spread_spectrum_steganography.decrypt(encrypted_image_array)
            rpe_hidden_text = RPEEncryptDecrypt().decode_rpe(rpe_img)

            # Function for safe writing with proper encoding
            def safe_write(file, text):
                try:
                    file.write(text)
                except UnicodeEncodeError:
                    # Handle non-ASCII characters by ignoring them
                    file.write(text.encode('ascii', 'ignore').decode())

            # Save decoded messages to text files with proper encoding and error handling
            with open("lsb_decoded.txt", "w", encoding="utf-8") as lsb_file:
                safe_write(lsb_file, lsb_hidden_text)
            with open("dct_decoded.txt", "w", encoding="utf-8") as dct_file:
                safe_write(dct_file, dct_hidden_text)
            with open("dwt_decoded.txt", "w", encoding="utf-8") as dwt_file:
                safe_write(dwt_file, dwt_hidden_text)
            with open("spread_spectrum_decoded.txt", "w", encoding="latin-1") as spread_spectrum_file:
                safe_write(spread_spectrum_file, decrypted_message)
            with open("rpe_decoded.txt", "w", encoding="latin-1") as rpe_file:
                safe_write(rpe_file, rpe_hidden_text)

            print("Decoded messages were saved in the Decoded_output folder.")
            os.chdir("..")

        elif m == "3":
            # Comparison Section
            os.chdir("Original_image/")
            original_image_file = input("Enter the name of the original file with extension : ")
            lsb_img = Image.open(original_image_file)
            dct_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
            dwt_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
            spread_spectrum_img = Image.open(original_image_file)
            os.chdir("..")
            os.chdir("Encoded_image/")
            lsb_img_encoded = Image.open(lsb_encoded_image_file)
            dct_img_encoded = cv2.imread(dct_encoded_image_file, cv2.IMREAD_UNCHANGED)
            dwt_img_encoded = cv2.imread(dwt_encoded_image_file, cv2.IMREAD_UNCHANGED)
            spread_spectrum_img_encoded = Image.open(spread_spectrum_encoded_image_file)
            rpe_img_encoded = Image.open(rpe_encoded_image_file)
            os.chdir("..")
            os.chdir("Comparison_result/")

            # Resize decoded images to match encoded images
            dct_img = cv2.resize(dct_img, (dct_img_encoded.shape[1], dct_img_encoded.shape[0]))

            # Comparison using different methods
            mse_lsb = Compare().meanSquareError(np.array(lsb_img), np.array(lsb_img_encoded))
            mse_dct = Compare().meanSquareError(dct_img, dct_img_encoded)
            mse_dwt = Compare().meanSquareError(dwt_img, dwt_img_encoded)
            mse_spread_spectrum = Compare().meanSquareError(np.array(spread_spectrum_img), np.array(spread_spectrum_img_encoded))
            mse_rpe = Compare().meanSquareError(np.array(lsb_img), np.array(rpe_img_encoded))

            psnr_lsb = Compare().psnr(np.array(lsb_img), np.array(lsb_img_encoded))
            psnr_dct = Compare().psnr(dct_img, dct_img_encoded)
            psnr_dwt = Compare().psnr(dwt_img, dwt_img_encoded)
            psnr_spread_spectrum = Compare().psnr(np.array(spread_spectrum_img), np.array(spread_spectrum_img_encoded))
            psnr_rpe = Compare().psnr(np.array(lsb_img), np.array(rpe_img_encoded))

            capacity_lsb = Compare().embedding_capacity(lsb_img.size[0] * lsb_img.size[1])
            capacity_dct = Compare().embedding_capacity(dct_img.size)
            capacity_dwt = Compare().embedding_capacity(dwt_img.size)
            capacity_spread_spectrum = Compare().embedding_capacity(spread_spectrum_img.size[0] * spread_spectrum_img.size[1])
            capacity_rpe = Compare().embedding_capacity(lsb_img.size[0] * lsb_img.size[1])

            correlation_lsb = Compare().correlation(np.array(lsb_img), np.array(lsb_img_encoded))
            correlation_dct = Compare().correlation(dct_img, dct_img_encoded)
            correlation_dwt = Compare().correlation(dwt_img, dwt_img_encoded)
            correlation_spread_spectrum = Compare().correlation(np.array(spread_spectrum_img), np.array(spread_spectrum_img_encoded))
            correlation_rpe = Compare().correlation(np.array(lsb_img), np.array(rpe_img_encoded))

            correlation_lsb_mean = np.mean(correlation_lsb)  # Calculate the mean of the correlation array
            correlation_dct_mean = np.mean(correlation_dct)
            correlation_dwt_mean = np.mean(correlation_dwt)
            correlation_spread_spectrum_mean = np.mean(correlation_spread_spectrum)
            correlation_rpe_mean = np.mean(correlation_rpe)

            workbook = xlwt.Workbook()
            sheet = workbook.add_sheet('Comparison')

            sheet.write(0, 0, 'Method')
            sheet.write(0, 1, 'MSE')
            sheet.write(0, 2, 'PSNR')
            sheet.write(0, 3, 'Correlation')
            sheet.write(0, 4, 'Embedding Capacity (bits)')

            sheet.write(1, 0, 'LSB')
            sheet.write(1, 1, mse_lsb)
            sheet.write(1, 2, psnr_lsb)
            sheet.write(1, 3, correlation_lsb_mean)
            sheet.write(1, 4, capacity_lsb)

            sheet.write(2, 0, 'DCT')
            sheet.write(2, 1, mse_dct)
            sheet.write(2, 2, psnr_dct)
            sheet.write(2, 3, correlation_dct_mean)
            sheet.write(2, 4, capacity_dct)

            sheet.write(3, 0, 'DWT')
            sheet.write(3, 1, mse_dwt)
            sheet.write(3, 2, psnr_dwt)
            sheet.write(3, 3, correlation_dwt_mean)
            sheet.write(3, 4, capacity_dwt)

            sheet.write(4, 0, 'Spread Spectrum')
            sheet.write(4, 1, mse_spread_spectrum)
            sheet.write(4, 2, psnr_spread_spectrum)
            sheet.write(4, 3, correlation_spread_spectrum_mean)
            sheet.write(4, 4, capacity_spread_spectrum)

            sheet.write(5, 0, 'RPE')
            sheet.write(5, 1, mse_rpe)
            sheet.write(5, 2, psnr_rpe)
            sheet.write(5, 3, correlation_rpe_mean)
            sheet.write(5, 4, capacity_rpe)

            workbook.save('Comparison_result.xls')

            print("Comparison result was saved in the Comparison_result folder.")
            os.chdir("..")

        else:
            break