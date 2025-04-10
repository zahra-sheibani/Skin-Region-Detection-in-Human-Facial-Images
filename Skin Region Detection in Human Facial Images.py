import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import time

# 1
df = pd.read_parquet("hf://datasets/NekoJojo/modified_wider_face_val/data/validation-00000-of-00001.parquet")

df['image_array'] = df['image'].apply(lambda x: cv2.imdecode(np.frombuffer(x['bytes'], np.uint8), cv2.IMREAD_COLOR))

# 2
features = []
labels = []

for idx, row in df.iterrows():
    image = row['image_array']
    label_list = row['labels']  


    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)


    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)


    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)


    for i in range(len(label_list)):
        row_idx = i // image.shape[1]
        col_idx = i % image.shape[1]
        features.append([
            h[row_idx, col_idx], s[row_idx, col_idx], v[row_idx, col_idx], 
            image[row_idx, col_idx, 0], image[row_idx, col_idx, 1], image[row_idx, col_idx, 2],  
            row_idx / image.shape[0], col_idx / image.shape[1],  
            edges[row_idx, col_idx]  
        ])
        labels.append(label_list[i])


features = np.array(features)
labels = np.array(labels)

# 3
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 4
start_time = time.time()
decision_tree = DecisionTreeClassifier(max_depth=10, min_samples_split=10, class_weight="balanced", random_state=42)
decision_tree.fit(X_train, y_train)
dt_train_time = time.time() - start_time

start_time = time.time()
y_pred_dt = decision_tree.predict(X_test)
dt_test_time = time.time() - start_time

print("result on decision tree")
print(classification_report(y_test, y_pred_dt))
print("accuracy:", accuracy_score(y_test, y_pred_dt))
print(f"training time: {dt_train_time:.2f} prediction time , sec: {dt_test_time:.2f} sec\n")

# 5
start_time = time.time()
random_forest = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
random_forest.fit(X_train, y_train)
rf_train_time = time.time() - start_time


start_time = time.time()
y_pred_rf = random_forest.predict(X_test)
rf_test_time = time.time() - start_time


print("Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("accuracy on Random Forest:", accuracy_score(y_test, y_pred_rf))
print(f"training time: {rf_train_time:.2f} prediction time , sec: {rf_test_time:.2f} sec\n")

# 6
def predict_skin_regions(image, model):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)

    features = np.stack([
        h.flatten(), s.flatten(), v.flatten(),
        image[:, :, 0].flatten(), image[:, :, 1].flatten(), image[:, :, 2].flatten(),
        np.repeat(np.arange(image.shape[0]) / image.shape[0], image.shape[1]),
        np.tile(np.arange(image.shape[1]) / image.shape[1], image.shape[0]),
        edges.flatten()
    ], axis=1)

    predictions = model.predict(features)

    skin_mask = predictions.reshape(image.shape[:2]) * 255  
    return skin_mask.astype(np.uint8)

sample_image = df.iloc[0]['image_array']
predicted_mask_dt = predict_skin_regions(sample_image, decision_tree)
predicted_mask_rf = predict_skin_regions(sample_image, random_forest)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('ORGINAL')
plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('decision tree predicted mask')
plt.imshow(predicted_mask_dt, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Random Forest predicted mask')
plt.imshow(predicted_mask_rf, cmap='gray')
plt.axis('off')

plt.show()
