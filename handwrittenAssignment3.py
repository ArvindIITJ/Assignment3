#!/usr/bin/env python
# coding: utf-8

# In[35]:


# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import pandas as pd
digits = datasets.load_digits()
data = digits.images.reshape((n_samples, -1))


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gamma_list=[0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list=[0.1 ,0.2 ,0.5 ,0.7 ,1 ,2 ,5 ,7 ,10]
#model hyper_parameters

train_frac=0.1
test_frac=0.1
dev_frac=0.1

# Split data into 50% train and 50% test subsets

X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=1-train_frac, shuffle=True
)
dev_test_frac=1-train_frac
X_train, X_dev, y_train, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


# In[37]:


digits = datasets.load_digits()
data = digits.images.reshape((n_samples, -1))
print("---Digits--")
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation="nearest")


# In[38]:


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# In[32]:


df =pd.DataFrame()

g = []
cc=[]
accuracy =[]

for GAMMA in gamma_list:
    for c in c_list:
        # Create a classifier: a support vector classifier
        clf = svm.SVC(gamma=GAMMA, C=c)
        
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_dev)
        
        score = accuracy_score(y_pred=predicted,y_true=y_dev)
        
        g.append(GAMMA)
        cc.append(c)
        accuracy.append(score)
        
        
df['Gamma'] = g
df['C']= cc
df['Accuracy'] = accuracy

df


# In[33]:


from PIL import Image
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    image_resized = resize(image, (int(image.shape[0] // 4), int(image.shape[1] // 2)),
                       anti_aliasing=True)
    


    ax.set_title(f"Prediction  : {prediction}" f"ImageSize: {image.size}")
    print("the image with the size" f"Prediction: {prediction}" f"image_resized :{image_resized}\n")


# In[34]:


a= df['Accuracy']
maximum = a.max()
index = a.idxmax()

print("The best test score is ", maximum," corresponding to hyperparameters gamma= ",g[index]," C=",cc[index])


# In[ ]:




