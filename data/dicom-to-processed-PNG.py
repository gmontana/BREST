#!/usr/bin/env python3
"""
Script to directly convert DICOM images (using path information from a CSV)
to final processed PNG images.

The script does the following:
  1. Reads the CSV file (which must contain at least a "path" column and optionally an "ImageLaterality" column).
  2. For each row, it constructs the full path to the DICOM file (by appending a '.dcm' extension)
     using the given base directory.
  3. Loads the DICOM image, converts the pixel data to an 8-bit format and creates a PIL image.
  4. Computes the aspect ratio and applies padding (and flipping) based on the "ImageLaterality" field.
  5. Resizes the image to the desired dimensions and saves it as a PNG, maintaining the folder structure.
  6. Updates the CSV with the computed aspect ratios and saves it in the output directory.

Usage Example (from the command line):
    python dicom_to_processed_png.py --csv CSVs/Oxford_CAD_1to3_V4.csv \
        --dicom-dir /path/to/dicom/base/directory \
        --output-dir /path/to/output/directory \
        --size 1792
"""

import argparse
import os
import pandas as pd
import pydicom
import numpy as np
from PIL import Image

def dicom_to_pil_image(dicom_file):
    """
    Reads a DICOM file and converts its pixel data into a PIL Image.
    
    This function:
      - Loads the DICOM file.
      - Converts the pixel data to float, normalises it to the range 0-255,
        and converts it to an 8-bit unsigned integer array.
      - Creates a PIL image from the 8-bit array and converts it into RGB.
    
    Parameters:
        dicom_file (str): Full path to the DICOM file.
    
    Returns:
        PIL.Image.Image or None: The converted PIL image, or None if an error occurred.
    """
    try:
        # Read the DICOM file.
        ds = pydicom.dcmread(dicom_file)
        # Convert the pixel array to a float type to avoid underflow/overflow issues.
        image_array = ds.pixel_array.astype(float)
        # Normalise the pixel values to the range 0-255.
        image_array_scaled = (np.maximum(image_array, 0) / np.max(image_array)) * 255.0
        # Convert the scaled pixel data to 8-bit unsigned integer.
        image_uint8 = np.uint8(image_array_scaled)
        # Create a PIL image from the array.
        pil_image = Image.fromarray(image_uint8)
        # Ensure the image is in RGB format.
        pil_image = pil_image.convert('RGB')
        return pil_image
    except Exception as e:
        print(f"Error converting DICOM {dicom_file}: {e}")
        return None

def process_image(row, dicom_dir, final_size):
    """
    Processes a single image based on CSV information.
    
    For the given row, it:
      - Constructs the DICOM file path.
      - Converts the DICOM file to a PIL image.
      - Computes the aspect ratio.
      - Applies padding and (if necessary) flipping based on the 'ImageLaterality' field.
      - Resizes the image to a square image of final_size x final_size.
    
    Parameters:
        row (pandas.Series): A row from the CSV containing at least the 'path' field
                             and optionally the 'ImageLaterality' field.
        dicom_dir (str): Base directory where the source DICOM images are stored.
        final_size (int): The width and height (in pixels) for the final PNG image.
    
    Returns:
        tuple or None:
            (processed PIL Image, computed aspect ratio) if successful, otherwise None.
    """
    # Construct the full DICOM file path by appending '.dcm' to the 'path' field.
    dicom_file = os.path.join(dicom_dir, row['path'] + '.dcm')
    
    if not os.path.exists(dicom_file):
        print(f"File not found: {dicom_file}")
        return None

    # Convert the DICOM image to a PIL image.
    img = dicom_to_pil_image(dicom_file)
    if img is None:
        return None

    # Compute the aspect ratio (width divided by height), rounded to two decimal places.
    aspect_ratio = round(img.width / img.height, 2)

    # Retrieve image laterality from the CSV row (if available).
    laterality = row.get('ImageLaterality', None)
    
    # If the image laterality is specified (either 'R' or 'L'), apply the padding strategy.
    if laterality in ['R', 'L']:
        # For these cases, the desired outcome is a square image with side length equal to the image height.
        new_width = img.height
        # Create a new black image (square) to accommodate the original image plus any padding.
        new_img = Image.new('RGB', (new_width, img.height), color='black')
        
        if laterality == 'L':
            # For left-oriented images, no horizontal padding is applied (paste at (0,0)).
            padding = (0, 0)
            new_img.paste(img, padding)
            # Crop is applied to ensure the image is exactly square (this may be redundant
            # if the new image has been created with the target dimensions).
            img = new_img.crop((0, 0, new_width, img.height))
        else:
            # For right-oriented images, calculate horizontal padding.
            # This pads the left side so that, when flipped, the padding appears on the right.
            padding = (img.height - img.width, 0)
            new_img.paste(img, padding)
            img = new_img
            # Flip the image horizontally.
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Resize the processed image to the final required dimensions.
    img = img.resize((final_size, final_size))
    return img, aspect_ratio

def main():
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(description="Directly convert DICOM images to final processed PNG images using CSV information.")
    parser.add_argument('--csv', required=True,
                        help="Path to the CSV file (e.g. CSVs/Oxford_CAD_1to3_V4.csv).")
    parser.add_argument('--dicom-dir', required=True,
                        help="Base directory containing the source DICOM images.")
    parser.add_argument('--output-dir', required=True,
                        help="Directory where the final processed PNG images will be saved.")
    parser.add_argument('--size', type=int, default=1792,
                        help="Final size (in pixels) for the square processed PNG image (default: 1792).")
    args = parser.parse_args()
    
    # Read the CSV file.
    df = pd.read_csv(args.csv)
    
    # Ensure that the CSV contains the required 'path' column.
    if 'path' not in df.columns:
        print("Error: CSV file must contain a column named 'path'.")
        return
    
    # If the CSV does not include 'ImageLaterality', create the column with default None values.
    if 'ImageLaterality' not in df.columns:
        df['ImageLaterality'] = None

    # Create the output directory if it does not already exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialise a new column in the DataFrame for aspect ratios.
    df['aspect_ratio'] = None

    # Process each row in the CSV.
    for index, row in df.iterrows():
        result = process_image(row, args.dicom_dir, args.size)
        if result is None:
            print(f"Skipping row {index} due to an error.")
            continue
        processed_img, aspect_ratio = result
        # Update the DataFrame with the computed aspect ratio.
        df.at[index, 'aspect_ratio'] = aspect_ratio
        # Construct the output file path (maintaining any subdirectory structure if desired).
        out_path = os.path.join(args.output_dir, row['path'] + '.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Save the processed image.
        processed_img.save(out_path)
        print(f"Processed image saved to: {out_path}")
    
    # Save the updated CSV (with aspect ratios) in the output directory.
    updated_csv = os.path.join(args.output_dir, "processed_" + os.path.basename(args.csv))
    df.to_csv(updated_csv, index=False)
    print(f"Updated CSV with aspect ratios saved to: {updated_csv}")

if __name__ == "__main__":
    main()
