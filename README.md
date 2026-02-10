# FarsiDigitw-synthetic
"Synthetic dataset of 30,000 Persian (Farsi) digits (64×64) with random fonts, positions, rotations, and pixel noise – designed for challenging ML/OCR generalization

# Introduction 📜
Hey! 👋 If you've worked with the famous MNIST dataset, you know it's great for practicing ML models on handwritten English digits. But why not have something similar for Persian/Arabic digits? After some searching, I found a few, but none were challenging enough. So, I decided to build one from scratch!
This project generates a synthetic dataset of Persian digits (0-9, excluding 8) using the PIL library. The goal? Create a truly tough dataset that forces models to generalize well. I created over 30,000 64x64 pixel images packed with variety: different fonts, random positions, rotations, and pixel noise!
This dataset is perfect for training classification models and could be a foundation for Persian OCR projects. 🚀
# Key Features of the Dataset ✨
I designed this dataset to be challenging and test ML algorithms thoroughly. Details below:
Random Digit Selection 🔢: Digits are randomly chosen from [0,1,2,3,4,5,6,7,9] (Note: 8 is intentionally omitted for focus).
Diverse Fonts 🖋️: Each image uses one of 13 random Persian/Arabic fonts (like BCompset, BTitrBd, etc.) to accustom models to various styles.
Variable Positioning 📍: Digits are placed in random positions (x: 8-16, y: 4-12) to avoid dependency on fixed locations.
Random Rotation 🔄: Images are rotated by a random angle between -15 to +15 degrees, mimicking real-world handwriting tilts!
Smart Pixel Noise 🌫️: Half the images get random pixels (80-150 range), the other half pure black (0) to build resistance to noisy conditions.
These features turn the dataset into a real playground for ML algorithms. 😎
# Differences from MNIST Dataset ⚖️
MNIST is awesome, but this Persian dataset has its own twists:
Image Size 📏: MNIST is 28x28 pixels, but here it's 64x64 (more details, higher complexity).
Digit Type ✍️: MNIST has handwritten English digits; this one has typed Persian/Arabic with font variety and noise.
Sample Count 🔢: MNIST has 70,000 samples; this has 30,000 (compact yet sufficient for training).
Extra Challenges 💪: Added rotations, noise, and random positions test generalization more rigorously.
Overall, it's tailored for right-to-left languages like Persian and complements MNIST well.
# Model Performance 📈
Before training, I reduced dimensions from 4,096 to about 2,327 using SparseRandomProjection (eps=0.2 to preserve structure).
Tested Models:
Logistic Regression: Precision ~91%
SVC: Precision ~89%
KNN: Precision ~94%
RandomForestClassifier 🌳: Best performer with 96% Precision (no bootstrap, random_state=41).
Confusion matrix and Precision-Recall/ROC curves are in the code (commented for easy execution). The model handles new images great – e.g., correctly predicted a rotated "4"!
# Installation & Setup 🛠️
Clone the repo: git clone https://github.com/yourusername/farsi-digits-dataset.git
Install dependencies: pip install -r requirements.txt (includes numpy, scikit-learn, PIL, and matplotlib).
Generate dataset: Run the main code to create Farsi_numeric.csv.
