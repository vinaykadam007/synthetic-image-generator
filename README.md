# Synthetic Image Generator
Synthetic Image Generator is a Python tool for creating synthetic fluorescence microscopy images of yeast cells, along with their labeled counterparts. This generator is ideal for creating datasets for training machine learning models in cell segmentation tasks, particularly in neuroscience and biological research.

<p align="center">
  <img src="misc/demodisplay_image.png" alt="Synthetic Image Example" width="400"/>
  <br>
  <em>Example of a Generated Fluorescence Image (1024 x 1024)</em>
</p>

## 🚀 Features

- **Customizable Image Generation**: Configure image size, number of images, cell count, fluorescence levels, and camera noise level.
- **Realistic Fluorescence Images**: Generates high-quality images suitable for training segmentation models.
- **Labeled Outputs**: Provides corresponding labeled images with distinct cell identifiers.

## 📥 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/vinaykadam007/synthetic-image-generator.git
cd synthetic-image-generator
pip install -r requirements.txt
```

## 🛠️ Usage

To generate synthetic images, execute the app.py file:

<p align="center">
  <img src="misc/ScreenshotGUI.png" alt="Synthetic Image Example" width="400"/>
  <br>
  <em>Simple Graphical User Interface</em>
</p>


## 🔧 Input Parameters

- Image width: Width of the image in pixels.
- Image height: Height of the image in pixels.
- Num of cells: Number of cells to simulate.
- Min & Max fluorescence_level: Intensity of the fluorescence signal.
- Cell size: Specifying minimum and maximum cell sizes.
- Camera noise level: Amount of noise to add to the image for realism.
  
## 📤 Output Files

Fluorescence Image: A .png file of the generated yeast cells.
Labeled Image: A .png file with labeled cells, where the background is 0 and cells are labeled with incrementing integers.

## 📚 Examples

Generate a 128x128 pixel image with 9 cells:

<p align="center">
  <img src="misc/Fluorescence.png" alt="Synthetic Image Example" width="400"/>
  <img src="misc/Label.png" alt="Synthetic Image Example" width="400"/>
  <br>
  <em>Generated Fluorescence and Label Image</em>
</p>

    
## 🤝 Contributing
Welcome all contributions! Please fork the repository, create a new branch, and submit a pull request. Ensure your code is well-documented and adheres to the project's coding standards.
<p align="center"> <img src="https://img.shields.io/github/forks/vinaykadam007/synthetic-image-generator?style=social" alt="Forks"> <img src="https://img.shields.io/github/stars/vinaykadam007/synthetic-image-generator?style=social" alt="Stars"> </p>
