from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
import numpy as np
import os
import cv2

training_images = []
training_labels = []
testing_images = []
testing_labels = []
labels = ['cardboard', 'glass']
dim = 100
testing_path = 'D:\\ml\\image classification\\dataset\\test_set\\test_set'
training_path = 'D:\\ml\\image classification\\dataset\\train_set\\train_set'

for i in labels:
    path = os.path.join(training_path, i)
    for j in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, j), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (dim, dim))
            training_images.append(img_array)
            if i == 'cardboard':
                training_labels.append([1, 0])
            else:
                training_labels.append([0, 1])
        except Exception as e:
            pass
for i in labels:
    path = os.path.join(testing_path, i)
    for j in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, j), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (dim, dim))
            testing_images.append(img_array)
            if i == 'cardboard':
                testing_labels.append([1, 0])
            else:
                testing_labels.append([0, 1])
        except Exception as e:
            pass

X_train = np.array(training_images)
y_train = np.array(training_labels)
X_test = np.array(testing_images)
y_test = np.array(testing_labels)

X_train = X_train.reshape(723, dim, dim, 1)
X_test = X_test.reshape(181, dim, dim, 1)

model = Sequential([
    # Conv2D(100, kernel_size=3, activation='relu', input_shape=(384, 512, 1)),
    Conv2D(100, kernel_size=3, activation='relu', input_shape=(dim, dim, 1)),
    Conv2D(80, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, shuffle=True)

raw = []
newcardboard = cv2.imread('newcardboard.jpg', cv2.IMREAD_GRAYSCALE)
newcardboard = cv2.resize(newcardboard, (dim, dim))
raw.append(newcardboard)
newglass = cv2.imread('newglass.jpg', cv2.IMREAD_GRAYSCALE)
newglass = cv2.resize(newglass, (dim, dim))
raw.append(newglass)

out = np.array(raw)
out = out.reshape(2, dim, dim, 1)
print('Actual: ')
cv2.imshow('IMAGE 1', newcardboard)
cv2.waitKey(0)
cv2.imshow('IMAGE 2', newglass)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Predicted: ')
result = model.predict(out)
for i in result:
    if i[0] > i[1]:
        print('Cardboard')
    else:
        print('Glass')

print('DONE')
