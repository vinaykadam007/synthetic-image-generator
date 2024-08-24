import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGridLayout)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, QIcon
from skimage import filters, draw
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import shutil


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class ImageGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowIcon(QIcon('icons/python.png'))
        self.setWindowTitle('Synthetic Image Generator')
        self.setFixedSize(500, 600)  # Fixed window size

        # Center the window on the screen
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.setGeometry(QRect(x, y, self.width(), self.height()))

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout(self.central_widget)

        # Parameters
        self.width_label = QLabel("Image Width:")
        self.width_slider = QSpinBox()
        self.width_slider.setRange(0, 2048)
        self.width_slider.setValue(128)

        self.height_label = QLabel("Image Height:")
        self.height_slider = QSpinBox()
        self.height_slider.setRange(0, 2048)
        self.height_slider.setValue(128)

        self.num_cells_label = QLabel("Number of Cells:")
        self.num_cells_slider = QSpinBox()
        self.num_cells_slider.setRange(1, 1000000)
        self.num_cells_slider.setValue(9)

        self.num_images_label = QLabel("Number of Images:")
        self.num_images_slider = QSpinBox()
        self.num_images_slider.setRange(1, 10000)
        self.num_images_slider.setValue(5)

        self.min_size_label = QLabel("Min Cell Size:")
        self.min_size_slider = QSpinBox()
        self.min_size_slider.setRange(1, 20)
        self.min_size_slider.setValue(5)

        self.max_size_label = QLabel("Max Cell Size:")
        self.max_size_slider = QSpinBox()
        self.max_size_slider.setRange(5, 50)
        self.max_size_slider.setValue(16)

        self.fl_min_label = QLabel("Min Fluorescence Level:")
        self.fl_min_slider = QDoubleSpinBox()
        self.fl_min_slider.setRange(0.0, 1.0)
        self.fl_min_slider.setSingleStep(0.1)
        self.fl_min_slider.setValue(0.4)

        self.fl_max_label = QLabel("Max Fluorescence Level:")
        self.fl_max_slider = QDoubleSpinBox()
        self.fl_max_slider.setRange(0.0, 1.0)
        self.fl_max_slider.setSingleStep(0.1)
        self.fl_max_slider.setValue(0.6)

        self.noise_label = QLabel("Camera Noise:")
        self.noise_slider = QDoubleSpinBox()
        self.noise_slider.setRange(0.0, 0.1)
        self.noise_slider.setSingleStep(0.01)
        self.noise_slider.setValue(0.01)

        self.generate_button = QPushButton("Generate Image")
        self.generate_button.setStyleSheet("min-height: 50px; background-color: green; color: white; font-size: 20px;")
        self.generate_button.clicked.connect(self.generate_image)

        # Image display labels
        self.image_label = QLabel("                Fluorescence Image")
        self.segmented_label = QLabel("                       Labeled Image")

        # Layout for inputs (placed in two columns)
        self.layout.addWidget(self.width_label, 0, 0)
        self.layout.addWidget(self.width_slider, 0, 1)
        self.layout.addWidget(self.height_label, 1, 0)
        self.layout.addWidget(self.height_slider, 1, 1)
        self.layout.addWidget(self.num_images_label, 2, 0)
        self.layout.addWidget(self.num_images_slider, 2, 1)
        self.layout.addWidget(self.num_cells_label, 3, 0)
        self.layout.addWidget(self.num_cells_slider, 3, 1)
        self.layout.addWidget(self.min_size_label, 4, 0)
        self.layout.addWidget(self.min_size_slider, 4, 1)
        self.layout.addWidget(self.max_size_label, 5, 0)
        self.layout.addWidget(self.max_size_slider, 5, 1)
        self.layout.addWidget(self.fl_min_label, 6, 0)
        self.layout.addWidget(self.fl_min_slider, 6, 1)
        self.layout.addWidget(self.fl_max_label, 7, 0)
        self.layout.addWidget(self.fl_max_slider, 7, 1)
        self.layout.addWidget(self.noise_label, 8, 0)
        self.layout.addWidget(self.noise_slider, 8, 1)
        # self.layout.addWidget(self.generate_button, 9, 0, 1, 2)
        self.layout.addWidget(self.generate_button, 10, 0, 1, 2)
        # Layout for image displays (occupying the remaining columns)
        self.layout.addWidget(self.image_label, 9, 0)
        self.layout.addWidget(self.segmented_label, 9, 1)

    def generate_image(self):
        shutil.rmtree(resource_path('generated_images/images'))
        shutil.rmtree(resource_path('generated_images/labels'))
        width = self.width_slider.value()
        height = self.height_slider.value()
        num_images = self.num_images_slider.value()
        num_cells = self.num_cells_slider.value()
        cell_size_range = (self.min_size_slider.value(), self.max_size_slider.value())
        fluorescence_level = (self.fl_min_slider.value(), self.fl_max_slider.value())
        camera_noise = self.noise_slider.value()

        output_dir = 'generated_images'
        generate_batch_images(num_images, output_dir, width, height, num_cells,
                              fluorescence_level, cell_size_range, camera_noise)

        # Display the first generated image and labeled image
        fluorescence_image_path = os.path.join(output_dir, 'demodisplay_image.png')
        labeled_image_path = os.path.join(output_dir, 'demodisplay_label.png')
        
        self.display_image(fluorescence_image_path, self.image_label)
        self.display_image(labeled_image_path, self.segmented_label)

    def display_image(self, image_path, label):
        image = QImage(image_path)
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))

def generate_fluorescence_image(width=128, height=128, num_cells=9, 
                                fluorescence_level=(0.4, 0.6), 
                                cell_size_range=(5, 16), 
                                camera_noise=0.01):
    image = np.zeros((height, width), dtype=np.float32)
    shapes = ['irregular_round', 'ellipse', 'round']
    
    for _ in range(num_cells):
        center_y = np.random.randint(0, height)
        center_x = np.random.randint(0, width)
        base_radius = np.random.randint(cell_size_range[0], cell_size_range[1])
        cell_shape = np.random.choice(shapes)
        
        if cell_shape == 'round':
            rr, cc = draw.disk((center_y, center_x), base_radius, shape=image.shape)
        elif cell_shape == 'irregular_round':
            theta = np.linspace(0, 2 * np.pi, 100)
            radius_variation = np.random.uniform(-0.2, 0.2, size=theta.shape)
            r = base_radius + base_radius * radius_variation
            rr, cc = draw.polygon(center_y + r * np.sin(theta), center_x + r * np.cos(theta), shape=image.shape)
        elif cell_shape == 'ellipse':
            rr, cc = draw.ellipse(center_y, center_x, base_radius, np.random.randint(base_radius // 2, base_radius), shape=image.shape)
        
        fluorescence = np.random.uniform(fluorescence_level[0], fluorescence_level[1])
        image[rr, cc] += fluorescence
    
    image = gaussian_filter(image, 1)
    image += np.random.normal(scale=camera_noise, size=image.shape)
    image = np.clip(image, 0, 1)
    return image

def apply_watershed_segmentation(image):
    fluorescence_image_8bit = (image * 255).astype(np.uint8)
    threshold_value = threshold_otsu(fluorescence_image_8bit)
    binary_image = (fluorescence_image_8bit > threshold_value) * 255

    kernel = np.ones((2, 2), np.uint8)
    binary_image = binary_image.astype('uint8')
    sure_bg = cv2.dilate(binary_image, kernel, iterations=10)
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 0)
    ret, sure_fg = cv2.threshold(dist_transform, 0.10 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # markers[markers == 1] = 0
    # Apply the watershed algorithm
    markers = cv2.watershed(cv2.cvtColor(fluorescence_image_8bit, cv2.COLOR_GRAY2BGR), markers)
    # markers[markers == 1] = 0
    # Mark the boundaries in the original image (optional)
    # fluorescence_image_8bit[markers == 1] = 0
    markers[markers == -1] = 0

    # Optionally adjust other segments if needed; the rest of the segments are naturally labeled incrementally
    markers -= 1 



    return markers

def generate_batch_images(num_images, output_dir, width=128, height=128, num_cells=9, 
                          fluorescence_level=(0.4, 0.6), cell_size_range=(5, 16), camera_noise=0.01):
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for i in range(1, num_images + 1):
        fluorescence_image = generate_fluorescence_image(width, height, num_cells, fluorescence_level, cell_size_range, camera_noise)
        output_image = apply_watershed_segmentation(fluorescence_image)
        if i == 1:
            fluorescence_filename = os.path.join(images_dir, f"fluorescence_image_{i:03d}.png")
            label_filename = os.path.join(labels_dir, f"labeled_image_{i:03d}.png")
            cv2.imwrite(fluorescence_filename, (fluorescence_image * 255).astype(np.uint8))
            cv2.imwrite(label_filename, output_image)

            display_dir = os.path.join(output_dir, 'demo')
            plt.figure()
            plt.imshow(output_image)
            plt.axis('off')
            plt.savefig(display_dir + 'display_label.png', bbox_inches='tight', pad_inches=0)

            plt.figure()
            plt.imshow(fluorescence_image, cmap = 'gray')
            plt.axis('off')
            plt.savefig(display_dir + 'display_image.png', bbox_inches='tight', pad_inches=0)
        else:
            fluorescence_filename = os.path.join(images_dir, f"fluorescence_image_{i:03d}.png")
            label_filename = os.path.join(labels_dir, f"labeled_image_{i:03d}.png")

            cv2.imwrite(fluorescence_filename, (fluorescence_image * 255).astype(np.uint8))
            cv2.imwrite(label_filename, output_image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = ImageGeneratorApp()
    ex.show()
    sys.exit(app.exec_())

