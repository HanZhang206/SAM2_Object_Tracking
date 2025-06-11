def load_image_sequence(image_dir):
    """
    Load a sequence of images from a directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        list: List of loaded images.
    """
    import os

    image_files = sorted(os.listdir(image_dir))
    images = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        images.append(image_path)

    return images

