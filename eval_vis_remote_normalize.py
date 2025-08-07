from PIL import Image
import numpy as np
import albumentations as A


if __name__ == "__main__":
    # Paths to saved features
    reference_image_path = "/home/***/codes/MultiViewGeo/Sample4Geo/stitched_image_scaled.jpg"  # Replace with your reference features path

    data_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    # Read the image using Pillow
    # Increase the max image pixels limit
    Image.MAX_IMAGE_PIXELS = None  # 设置为 None 表示移除限制
    img = Image.open(reference_image_path)

    # Convert image to RGB (Pillow uses RGB by default, but this ensures it's consistent)
    img = img.convert("RGB")

    # Convert the image to a numpy array (Pillow's output is HxWxC, which matches OpenCV's BGR->RGB conversion)
    img = np.array(img)

    # Apply normalization using albumentations
    transform = A.Compose([
        A.Normalize(mean=mean, std=std)
    ])
    img = transform(image=img)["image"]

    # Convert normalized image back to uint8 for saving
    # Clip the values to the range [0, 1], rescale to [0, 255], and convert to uint8
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    # Convert back to Pillow Image and save
    img = Image.fromarray(img)
    img.save("/home/***/codes/MultiViewGeo/Sample4Geo/stitched_image_scaled_normalize.jpg")

