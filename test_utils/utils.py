import matplotlib.pyplot as plt
from PIL import Image


def show_img(img):
    """Show image using matplotlib.
    Args:
        img (np.ndarray | PIL.Image | str): Image to show.
    """
    if isinstance(img, str):
        img = Image.open(img)
    plt.imshow(img)
    plt.show()
