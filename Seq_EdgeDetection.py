#Edge Detection handling file.

if __name__ == "__main__":
    sys.exit(0)

import cv2
import sys
import numpy as np

def Edge_Get(source):
    img = cv2.imread(source)
    if img is not None:
        img_height, img_width, channels = img.shape
        img_gray = cv2.imread(source, cv2.IMREAD_GRAYSCALE)

        # Apply Sobel operator
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

        # Compute gradient magnitude
        gradient_magnitude = cv2.magnitude(sobelx, sobely)

        # Convert to uint8
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
        gm = np.zeros(gradient_magnitude.shape, gradient_magnitude.dtype)

        #Edge De-amplification
        for y in range(gradient_magnitude.shape[0]):
            gm[y] = np.clip(0.25*gradient_magnitude[y], 0, 255)

        # Display result
        gradient_magnitude_2 = cv2.cvtColor(gm, cv2.COLOR_GRAY2RGB)
        img = cv2.bitwise_or(img,gradient_magnitude_2)

        #cv2.imshow("Laplacian Edge Detection", img)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return img
    return None
