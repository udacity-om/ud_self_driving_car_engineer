
# coding: utf-8

# In[1]:


# Import the required modules
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Convolution2D, MaxPooling2D, Cropping2D, Dropout
import matplotlib.pyplot as plt

# Plot inline
get_ipython().magic('matplotlib inline')


# In[2]:


# Set constant parameters
batch_size = 32
data_path = 'data'
img_folder = 'IMG'
csv_file_name = 'driving_log.csv'

crop_top = 60
crop_bottom = 130
crop_left = 0
crop_right = 320

filter1_kernel_size = 5
filter2_kernel_size = 3
image_shape = (64, 64, 3)


# In[3]:


# Read data from csv files
csv_full_path = data_path + '/'+ csv_file_name
img_path = data_path + '/' + img_folder
print(csv_full_path, img_path)


# In[4]:


# Put each line from csv to a list
samples = []
with open(csv_full_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# In[5]:


print(samples[0])


# In[6]:


# Remove the first line from list as it conatins column header
del samples[0]
print(len(samples))
print(samples[0])


# In[7]:


# Split available data into 80% training and 20% validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[8]:


#name = data_path + '/' + samples[10][0].split('/')[-1]
#print(name)
#center_image = cv2.imread(name)
#plt.imshow(center_image)
#print(center_image.shape)


# In[9]:


#cropped_image = center_image[crop_top:crop_bottom, crop_left:crop_right]
#plt.imshow(cropped_image)
#print(cropped_image.shape)


# In[10]:


#r = 64.0 / cropped_image.shape[1]
#dim = (64, int(center_image.shape[0] * r))
#dim = (64, 64)

# perform the actual resizing of the image and show it
#resized = cv2.resize(cropped_image, dim, interpolation = cv2.INTER_AREA)
#plt.imshow(resized)
#print(resized.shape)


# In[11]:


def generator(samples, batch_size=32):
    # make the number of samples a multiple of batch_size
    N = (len(samples)//batch_size)*batch_size  

    X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)    
    
    # Loop forever so the generator never terminates
    while 1: 
        shuffle(samples)
        for offset in range(0, N, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            for j,batch_sample in enumerate(batch_samples):
                # Choose image randomly from Center/Left/Right camera
                img_pos_choice = np.random.choice([0,1,2])
                # Get the pth name of the image
                name = data_path + '/'+ batch_sample[img_pos_choice].split('/')[-1]
                #print(name)
                # Read image in BGR format
                image = cv2.imread(name)
                # Convert image to RGB format as drive.py uses RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                # Crop the image so as to have only road data 
                cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
                # Resize the image to reduce training time
                resized_image = cv2.resize(cropped_image, (64,64))
                
                # Add correction to steering angle based on choosen camera image.
                if img_pos_choice == 0:
                    angle = float(batch_sample[3])

                if img_pos_choice == 1:
                    angle = float(batch_sample[3]) + 0.2

                if img_pos_choice == 2: 
                    angle = float(batch_sample[3]) - 0.25
                
                # Flip random image
                flip_prob = np.random.random()
                if flip_prob > 0.5:
                    # Flip the image and reverse the steering angle
                    angle = -1*angle
                    resized_image = cv2.flip(resized_image, 1)

                X_batch[j] = resized_image
                y_batch[j] = angle

            yield X_batch, y_batch


# In[12]:


# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# In[13]:


# Initialize the model
model = Sequential()
# Pre-processing: Normalizing and mean centered
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = image_shape))
# Modified LeNet model
model.add(Convolution2D(nb_filter = 6, nb_row = filter1_kernel_size, nb_col = filter1_kernel_size, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(nb_filter = 6, nb_row = filter1_kernel_size, nb_col = filter1_kernel_size, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1000, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(120, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(84, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1))

# Compile the model with Adam optimizer and MSE loss method
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch =
    (len(train_samples)//batch_size)*batch_size, validation_data = validation_generator, 
    nb_val_samples = len(validation_samples), nb_epoch=8, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Save the model
model.save('model.h5')


# In[ ]:




