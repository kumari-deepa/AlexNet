
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense

Classifier=Sequential()#making pipeline for data to perform whatever conv,pooling
Classifier.add(Conv2D(96,(100,100),input_shape = (1000, 1000, 3),activation='relu',strides=(4,4),padding='valid'))
Classifier.add(MaxPooling2D(pool_size = (50,50),strides=(2,2),padding='valid'))
Classifier.add(BatchNormalization())
#2 convolution layer
Classifier.add(Conv2D(256,(5,5),input_shape = (1000,1000,3),activation='relu',strides=(2,2),padding='valid'))
Classifier.add(MaxPooling2D(pool_size = (5,5),strides=(2,2),padding='valid'))
Classifier.add(BatchNormalization())
#3 convolution layer
Classifier.add(Conv2D(384,(3,3),input_shape = (1000,1000,3),activation='relu',padding='valid'))
#4 convolution layer
Classifier.add(Conv2D(384,(3,3),input_shape = (1000,1000,3),activation='relu',padding='valid'))
#5 convolution layer
Classifier.add(Conv2D(256,(3,3),input_shape = (1000,1000,3),activation='relu',padding='valid'))
Classifier.add(MaxPooling2D(pool_size = (2,2),strides=(2,2)))

Classifier.add(Dense(units = 64, activation = 'relu'))
Classifier.add(Dense(units = 1,activation = 'relu'))
Classifier.add(Dense(units = 1,activation = 'relu'))
#Classifier.add(Dense(units = 1,activation = 'relu'))

Classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
#make image b/w ,how much you want to zoom or shear an image
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/Users/kumar/Desktop/dog_cat/images',target_size = (1000,1000),batch_size = 32,class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/Users/kumar/Desktop/dog_cat/images',target_size = (1000,1000),batch_size = 32,class_mode = 'binary')


model = Classifier.fit_generator(training_set,
                         steps_per_epoch = 800,
                         epochs = 5,
                         validation_data = test_set,    
                         validation_steps = 2000)

Classifier.save("model.h5")
print("Saved model to disk")

# Part 3 - Making new predictions




import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C:\Users\kumar\Desktop\dog_cat\images', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'baby'
    print(prediction)
elif result[0][1] == 1:
    prediction = 'cat'
    print(prediction)
elif result[0][2] == 1:
    prediction = 'flower'
    print(prediction)
elif result[0][3] == 1:
    prediction = 'girl'
    print(prediction)
elif result[0][4] == 1:
    prediction = 'map'
    print(prediction)
elif result[0][5] == 1:
    prediction = 'mumma'
    print(prediction)
elif result[0][6] == 1:
    prediction = 'orange'
    print(prediction)
elif result[0][7] == 1:
    prediction = 'sea'
    print(prediction)
elif result[0][8] == 1:
    prediction = 'tajmahal'
    print(prediction)