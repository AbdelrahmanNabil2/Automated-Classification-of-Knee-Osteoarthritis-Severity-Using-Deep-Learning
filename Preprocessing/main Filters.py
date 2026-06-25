import cv2
import numpy as np
import os
from pathlib import Path


def contrast_stretch(image):
    img_float = image.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)

    if max_val > min_val:
        stretched = 255 * (img_float - min_val) / (max_val - min_val)
    else:
        stretched = img_float

    return stretched.astype(np.uint8)


def histogram_equalization(image):
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)

    return cv2.LUT(image, table)


def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    return sharpened


def apply_filters(input_folder, output_folder, gamma_value=1.2):
    input_path = Path(input_folder)
    output_path = Path(output_folder)


    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return


    output_path.mkdir(parents=True, exist_ok=True)


    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in '{input_folder}'!")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Output folder: {output_folder}\n")


    filters = {
        'contrast_stretch': contrast_stretch,
        'histogram_equalization': histogram_equalization,
        'gamma_correction': lambda img: gamma_correction(img, gamma=gamma_value),
        'sharpen': sharpen_image
    }


    total_processed = 0
    for idx, image_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_file.name}")


        image = cv2.imread(str(image_file))

        if image is None:
            print(f"  ⚠️  Warning: Could not read image, skipping...")
            continue


        filename_base = image_file.stem
        extension = image_file.suffix


        processed_image = image.copy()

        print(f"  Applying filters:")
        for filter_name, filter_func in filters.items():
            processed_image = filter_func(processed_image)
            print(f"    ✓ {filter_name}")


        output_filename = f"{filename_base}_all_filters{extension}"
        output_path_file = output_path / output_filename
        cv2.imwrite(str(output_path_file), processed_image)
        print(f"  ✓ Saved as: {output_filename}")

        total_processed += 1
        print()

    print(f"{'='*60}")
    print(f"✓ Processing complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Total output images: {total_processed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    INPUT_FOLDER = "/Users/abdelrahmannabil/Desktop/preprocessing/sr_images/grade2"
    OUTPUT_FOLDER = "/Users/abdelrahmannabil/Desktop/preprocessing/SR_filtered/grade2"
    GAMMA_VALUE = 1.5

    print("="*60)
    print("IMAGE FILTER APPLICATION")
    print("="*60)
    print()
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Gamma value: {GAMMA_VALUE}")
    print()
    print("Starting processing...")
    print()


    apply_filters(INPUT_FOLDER, OUTPUT_FOLDER, GAMMA_VALUE)
