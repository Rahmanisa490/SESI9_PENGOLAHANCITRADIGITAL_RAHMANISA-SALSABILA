import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def roberts_edge_detection(image):
    
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])

    
    edge_x = convolve(image, roberts_x)
    edge_y = convolve(image, roberts_y)

    
    magnitude = np.hypot(edge_x, edge_y)
    magnitude = np.clip(magnitude, 0, 255)
    
    return magnitude


def sobel_edge_detection(image):
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    
    edge_x = convolve(image, sobel_x)
    edge_y = convolve(image, sobel_y)

    
    magnitude = np.hypot(edge_x, edge_y)
    magnitude = np.clip(magnitude, 0, 255)
    
    return magnitude


image = imageio.imread('image.png')  
if image.ndim == 3:
    image = np.mean(image, axis=2)  


edges_roberts = roberts_edge_detection(image)
edges_sobel = sobel_edge_detection(image)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(edges_roberts, cmap='gray')
axes[1].set_title('Edges using Roberts')
axes[1].axis('off')

axes[2].imshow(edges_sobel, cmap='gray')
axes[2].set_title('Edges using Sobel')
axes[2].axis('off')

plt.tight_layout()
plt.show()


imageio.imwrite('edges_roberts.jpg', edges_roberts.astype(np.uint8))
imageio.imwrite('edges_sobel.jpg', edges_sobel.astype(np.uint8))
