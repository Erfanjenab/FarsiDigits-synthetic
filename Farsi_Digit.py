import random
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, precision_recall_curve, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# Load and preprocess a sample image for demonstration
sample_image = Image.open("numbers/number6.jpg").convert("L")
resized_sample = sample_image.resize((64,64), Image.Resampling.LANCZOS)
sample_array = np.array(resized_sample).flatten().reshape(1,-1)

# Plot sample of pictures (commented out for reference)
# plt.figure()
# plt.imshow(sample_array.reshape(64,64), "grey")
# plt.title("Handwritten number of 3")
# plt.savefig("Handwritten number of 3.png")
# plt.show()

# List of Persian/Arabic fonts to use for generating synthetic digits
fonts = ["BBadr.ttf", "BCompset.ttf", "BTitrBd.ttf", "BBardiya.ttf", "BFarnaz.ttf", "BFerdosi.ttf", "BHoma.ttf", "BJadidBd.ttf",
         "BNazanin.ttf", "Mj_Bita Bold.TTF", "BKarim.ttf", "BRoya.ttf", "BKoodkBd.ttf"]

# Digits to generate (note: 8 is missing from the list)
digits = [0,1,2,3,4,5,6,7,9]

features = []
labels = []

random.seed(41)
# Generate synthetic dataset: Create 30,000 images of random digits with variations in font, position, angle, and noise
for i in range(30_000):
    random_font = random.choice(fonts)
    random_digit = random.choice(digits)
    x_position = random.randint(8,16)
    y_position = random.randint(4,12)
    noise_mode = random.randint(0,1)  # 0: add random noise, 1: set to pure black
    rotation_angle = random.randint(-15, 15)

    base_image = Image.new(color=255, mode="L", size=(64,64))
    draw_context = ImageDraw.Draw(base_image)

    selected_font = ImageFont.truetype(size=45, font=f"fonts/{random_font}")
    draw_context.text(xy=(x_position, y_position), font=selected_font, fill=0, text=str(random_digit))
    rotated_image = base_image.rotate(angle=rotation_angle, fillcolor=255)

    image_array = np.array(rotated_image).flatten()

    if noise_mode == 0:
        mask = image_array < 255
        image_array[mask] = np.random.randint(80,150, size=image_array.shape)[mask]
    else:
        mask = image_array < 255
        image_array[mask] = 0
    
    features.append(image_array)
    labels.append(random_digit)

# Plot pictures in features dataset (commented out for reference)
# plt.figure()
# plt.imshow(sample_array.reshape(64,64), "grey")
# plt.show()

# plt.figure(figsize=(19.2,10.8))
# k=1
# for i in np.arange(3600, 3636, 1):
#     plt.subplot(6,6,k)
#     plt.imshow(features[i].reshape(64,64), "grey")
#     k += 1

# plt.savefig("numbers.png")
# plt.show()

# Split dataset into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, test_size=0.2)

# Create binary labels for digit 3 (used in precision-recall and ROC curves)
labels_train_is_3 = []
for idx, label in enumerate(labels_train):
    is_three = labels_train[idx] == 3
    labels_train_is_3.append(is_three)

# Dimensionality reduction using Sparse Random Projection to preserve structure with epsilon=0.2
dim_reducer = SparseRandomProjection(eps=0.2, random_state=41, dense_output=False, density="auto")
features_train_reduced = dim_reducer.fit_transform(features_train)
features_test_reduced = dim_reducer.transform(features_test)

# Test and fine-tune models (other models commented out; focusing on RandomForestClassifier)
# LogisticRegression 
# log_reg = LogisticRegression(n_jobs=-1)
# log_reg.fit(features_train_reduced, labels_train)

# Support vector classifier 
# svc = SVC()
# svc.fit(features_train_reduced[:1000], labels_train[:1000])

# K-nearest
# knn = KNeighborsClassifier()
# knn.fit(features_train_reduced, labels_train)

# RandomForestClassifier 
forest_clf = RandomForestClassifier(n_jobs=-1, bootstrap=False, random_state=41)
forest_clf.fit(features_train_reduced, labels_train)

# Cross-validation predictions and evaluation (commented out; results noted for reference)
# cross_predictions = cross_val_predict(forest_clf, features_train_reduced, labels_train, n_jobs=-1, cv=3)
# eval_scores = cross_val_score(forest_clf, features_train_reduced, labels_train, n_jobs=-1, cv=4, scoring="precision_micro")
# print(f"cross validation score (Precision): {eval_scores.mean()}")   # log_reg: 0.91
#                                                                    # svc: 0.89
#                                                                    # knn: 0.94
#                                                                    # forest_clf: 0.96

# precision_forest, recall_forest, thresholds_pr_forest = precision_recall_curve(labels_train_is_3, cross_predictions)
# fpr_forest, tpr_forest, thresholds_roc_forest = roc_curve(labels_train_is_3, cross_predictions)

# Plot precision and recall vs thresholds (commented out)
# plt.figure()
# plt.plot(thresholds_pr_forest, precision_forest[:-1], "b-", linewidth=3, label="Precision")
# plt.plot(thresholds_pr_forest, recall_forest[:-1], "r-", linewidth=3, label="Recall")
# plt.title("Precision and recall with thresholds on number 3 (Random Forest)")
# plt.legend()
# plt.grid()
# plt.savefig("Precision and recall with thresholds on number3 (Randomforest).png")

# Plot precision-recall curve (commented out)
# plt.figure()
# plt.plot(recall_forest, precision_forest, "b-", linewidth=4, label="Recall per Precision")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Recall per Precision on number 3 (Random Forest)")
# plt.grid()
# plt.legend()
# plt.savefig("Recall per precision on number3 (Randomforest).png")

# Plot ROC curve (commented out)
# plt.figure()
# plt.plot(fpr_forest, recall_forest, "b-", linewidth=4, label="FPR per Recall")
# plt.xlabel("FPR")
# plt.ylabel("Recall")
# plt.title("FPR per Recall on number 3 (Random Forest)")
# plt.grid()
# plt.legend()
# plt.savefig("FPR per Recall on number3 (knn).png")  # Note: Original had 'knn' typo; corrected to Random Forest in title
# plt.show()

# Plot confusion matrix (commented out)
# ConfusionMatrixDisplay.from_predictions(y_true=labels_train, y_pred=cross_predictions, cmap="Blues")
# plt.savefig("Confusion_matrix_(Randomforest).png")
# plt.show()

# Create a new instance image for testing the model
test_image = Image.new(color=255, size=(64,64), mode="L")
test_draw = ImageDraw.Draw(test_image)

test_font = ImageFont.truetype(font="fonts/BCompset.ttf", size=45)
test_draw.text(xy=(12, 7), text="4", font=test_font, fill=0)
rotated_test = test_image.rotate(5, fillcolor=255)

test_array = np.array(rotated_test).flatten().reshape(1,-1)

# plt.figure()
# plt.imshow(test_array.reshape(64,64), "gray")
# plt.show()

# Test model with new instance
test_reduced = dim_reducer.transform(test_array)
predicted_label = forest_clf.predict(test_reduced)
print(predicted_label)

# Convert datasets to CSV (commented out for reference)
# df = pd.DataFrame({"x": features,
#                    "y": labels})
# df.to_csv("Farsi_numeric.csv")