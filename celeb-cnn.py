
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# Setup directories
image_dir='cropped'
messi_imgs=os.listdir(image_dir+ '/lionel_messi')
sharapova_imgs=os.listdir(image_dir+ '/maria_sharapova')
federer_imgs=os.listdir(image_dir+ '/roger_federer')
williams_imgs=os.listdir(image_dir+ '/serena_williams')
kohli_imgs=os.listdir(image_dir+ '/virat_kohli')
print("--------------------------------------\n")

print('The length of LIONEL MESSI images is',len(messi_imgs))
print('The length of MARIA SHARAPOVA images is',len(sharapova_imgs))
print('The length of ROGER FEDERER images is',len(federer_imgs))
print('The length of SERENA WILLIAMS images is',len(williams_imgs))
print('The length of VIRAT KOHLI images is',len(kohli_imgs))
print("--------------------------------------\n")


# Prep dataset
dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(messi_imgs),desc="Lionel Messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append('messi')
        
        
for i ,image_name in tqdm(enumerate(sharapova_imgs),desc="Maria Sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append('sharapova')


for i , image_name in tqdm(enumerate(federer_imgs),desc="Roger Federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append('federer')


for i , image_name in tqdm(enumerate(williams_imgs),desc="Serena Williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append('williams')


for i , image_name in tqdm(enumerate(kohli_imgs),desc="Virat Kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append('kohli')
        
        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(label)

print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,Y,test_size=0.25,random_state=42, stratify=Y)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalising the Dataset. \n")

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

shape = x_train.shape[1:]
print("--------------------------------------\n")


# Since the task is image classification, CNN is the most appropriate neural network architecture.
# Training CNN Model
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])

print("--------------------------------------\n")
model.summary()
print("--------------------------------------\n")

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


print("--------------------------------------\n")
print("Training Started.\n")
history=model.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.1)
print("Training Finished.\n")
print("--------------------------------------\n")


# Plot and save accuracy
plt.plot(history.epoch,history.history['accuracy'], label='accuracy')
plt.plot(history.epoch,history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('results/celeb_acc_plot2.png')

# Clear the previous plot
plt.clf()

# Plot and save loss
plt.plot(history.epoch,history.history['loss'], label='loss')
plt.plot(history.epoch,history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('results/celeb_loss_plot2.png')


print("--------------------------------------\n")
print("Model Evaluation Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print('classification Report\n', classification_report(y_test,y_pred_classes))
print("--------------------------------------\n")

model.save("celeb-cnn-model.h5")
