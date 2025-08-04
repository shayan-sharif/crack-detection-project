import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img, binary

def morphological_processing(binary_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return morph

def edge_detection(morph_img):
    edges = cv2.Canny(morph_img, 50, 150)
    return edges

def detect_lines(edges, orig_img):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    img_lines = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(img_lines, (x1,y1), (x2,y2), (0,0,255), 2)
    return img_lines, lines

def main():
    img_path = r'F:\new_project\industrial_crack_detection\images\input\metal-texture-with-dust-scratches-and-cracks-photo.jpg'  
    orig_img, binary_img = preprocess_image(img_path)
    morph_img = morphological_processing(binary_img)
    edges = edge_detection(morph_img)
    img_with_lines, lines = detect_lines(edges, orig_img)

    print(f"Detected {0 if lines is None else len(lines)} cracks/lines.")

    plt.figure(figsize=(12,8))
    plt.subplot(1,3,1), plt.title('Original Image'), plt.imshow(orig_img, cmap='gray'), plt.axis('off')
    plt.subplot(1,3,2), plt.title('Edges'), plt.imshow(edges, cmap='gray'), plt.axis('off')
    plt.subplot(1,3,3), plt.title('Detected Lines'), plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()

