from PIL import Image

def text_to_binary(message):
    return ''.join(format(ord(char), '08b') for char in message)

def encode_lsb(original_image_path, secret_message, output_image_path):
    img = Image.open(original_image_path)
    binary_message = text_to_binary(secret_message)

    if len(binary_message) > img.size[0] * img.size[1] * 3:
        raise ValueError("Message is too long to be encoded in the image.")

    pixels = list(img.getdata())
    
    # Ensure the message length matches the pixel count
    binary_message = binary_message.ljust(len(pixels)*3, '0')  

    for i in range(len(pixels)):
        pixel_value = list(pixels[i])
        for j in range(3):  # RGB channels
            pixel_value[j] = int(format(pixel_value[j], '08b')[:-1] + binary_message[i*3+j], 2)
        pixels[i] = tuple(pixel_value)

    new_img = Image.new(img.mode, img.size)
    new_img.putdata(pixels)
    new_img.save(output_image_path)

# Example usage for encryption
original_image_path = r'C:\Users\harsh\OneDrive\Desktop\demo.jpg'
secret_message = 'hello harsh'
output_image_path = r'C:\Users\harsh\OneDrive\Desktop\output.jpg'

encode_lsb(original_image_path, secret_message, output_image_path)
