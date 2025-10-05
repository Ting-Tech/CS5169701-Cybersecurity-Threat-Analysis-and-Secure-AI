import os
import numpy as np
from PIL import Image

def convert_to_grayscale_image(input_folder, output_folder):
    for parent_dir in os.listdir(input_folder):
        parent_path = os.path.join(input_folder, parent_dir)
        if not os.path.isdir(parent_path):
            continue

        # establish output directory
        output_parent_path = os.path.join(output_folder, parent_dir)
        os.makedirs(output_parent_path, exist_ok=True)

        # deliver files
        for file_name in os.listdir(parent_path):
            file_path = os.path.join(parent_path, file_name)
            if not os.path.isfile(file_path):
                continue

            with open(file_path, 'rb') as f:
                byte_content = f.read()

            byte_array = np.frombuffer(byte_content, dtype=np.uint8)
            image_side = int(np.ceil(np.sqrt(len(byte_array))))
            # add padding if necessary
            padded_array = np.pad(byte_array, (0, image_side * image_side - len(byte_array)), mode='constant')
            gray_image_array = padded_array.reshape((image_side, image_side))
            gray_image = Image.fromarray(gray_image_array, 'L')

            output_image_path = os.path.join(output_parent_path, f"{os.path.splitext(file_name)[0]}.png")
            gray_image.save(output_image_path)
            print(f"✅ Saved {output_image_path}")

# main function
if __name__ == "__main__":
    input_folder = r"C:\Users\allen\.conda\envs\virus_pic\gray_virus\PEs"
    output_folder = r"C:\Users\allen\.conda\envs\virus_pic\gray_virus\PEs_gray"
    os.makedirs(output_folder, exist_ok=True)
    convert_to_grayscale_image(input_folder, output_folder)
