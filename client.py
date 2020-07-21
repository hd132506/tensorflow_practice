import socket
import numpy as np
import os
import gc
from scipy import misc
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
GPU Setting
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


"""
Data Loader
"""
def loadData(pathToFile, oneHot=False):
    """
    pathToDatasetFolder: Parent folder of CINIC-10 dataset folder or CINIC-10.tar.gz file
    oneHot: Label encoding (one hot encoding or not)
    Return: Train, validation and test sets and label numpy arrays
    """
    labelDict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                'truck': 9}

    pathToTrain = os.path.join(pathToFile, "train")
    pathToVal = os.path.join(pathToFile, "valid")
    pathToTest = os.path.join(pathToFile, "test")

    imgNamesTrain = [f for dp, dn, fn in os.walk(os.path.expanduser(pathToTrain)) for f in fn]
    imgDirsTrain = [dp for dp, dn, fn in os.walk(os.path.expanduser(pathToTrain)) for f in fn]
    imgNamesVal = [f for dp, dn, fn in os.walk(os.path.expanduser(pathToVal)) for f in fn]
    imgDirsVal = [dp for dp, dn, fn in os.walk(os.path.expanduser(pathToVal)) for f in fn]
    imgNamesTest = [f for dp, dn, fn in os.walk(os.path.expanduser(pathToTest)) for f in fn]
    imgDirsTest = [dp for dp, dn, fn in os.walk(os.path.expanduser(pathToTest)) for f in fn]

    XTrain = np.empty((len(imgNamesTrain), 32, 32, 3), dtype=np.float32)
    YTrain = np.empty((len(imgNamesTrain)), dtype=np.int32)
    XVal = np.empty((len(imgNamesVal), 32, 32, 3), dtype=np.float32)
    YVal = np.empty((len(imgNamesVal)), dtype=np.int32)
    XTest = np.empty((len(imgNamesTest), 32, 32, 3), dtype=np.float32)
    YTest = np.empty((len(imgNamesTest)), dtype=np.int32)

    print("Loading")

    for i in range(len(imgNamesTrain)):
        img = plt.imread(os.path.join(imgDirsTrain[i], imgNamesTrain[i]))
        if len(img.shape) == 2:
            XTrain[i, :, :, 2] = XTrain[i, :, :, 1] = XTrain[i, :, :, 0] = img/255.
        else:
            XTrain[i] = img/255.
        YTrain[i] = labelDict[os.path.basename(imgDirsTrain[i])]
        
    print(1)
    
    for i in range(len(imgNamesVal)):
        img = plt.imread(os.path.join(imgDirsVal[i], imgNamesVal[i]))
        if len(img.shape) == 2:
            XVal[i, :, :, 2] = XVal[i, :, :, 1] = XVal[i, :, :, 0] = img/255.
        else:
            XVal[i] = img/255.
        YVal[i] = labelDict[os.path.basename(imgDirsVal[i])]
        
    print(2)
    
    for i in range(len(imgNamesTest)):
        img = plt.imread(os.path.join(imgDirsTest[i], imgNamesTest[i]))
        if len(img.shape) == 2:
            XTest[i, :, :, 2] = XTest[i, :, :, 1] = XTest[i, :, :, 0] = img/255.
        else:
            XTest[i] = img/255.
        YTest[i] = labelDict[os.path.basename(imgDirsTest[i])]

    print(3)
        
    if oneHot:
        YTrain = toOneHot(YTrain, 10)
        YVal = toOneHot(YVal, 10)
        YTest = toOneHot(YTest, 10)

    print("+ Dataset loaded")

    return XTrain, YTrain, XVal, YVal, XTest, YTest

XTrain, YTrain, XVal, YVal, XTest, YTest = loadData("./cinic")

x = np.concatenate((XTrain, XVal, XTest)) * 255
y = np.concatenate((YTrain, YVal, YTest))

data_size = x.shape[0]
indice = np.arange(data_size)
np.random.shuffle(indice)


"""
Parameters about dataset
"""
test_class_n = 30000
test_idc = indice[:test_class_n]

x_test, y_test = x[test_idc], y[test_idc]

indice = indice[test_class_n:]

data_size = len(indice)

ratio = 0.8
split = int(data_size * ratio)

t = indice[:split]
x_train, y_train = x[t], y[t]

f = indice[split:]
x_fine, y_fine = x[f], y[f]

batch_size = 64
n_epochs = 300

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    brightness_range=[0.5, 1.5]
)

datagen.fit(x_train)


"""
Test setting
"""
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

@tf.function
def test_step(images, labels, model):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


"""
Loss obj & Optimizers
"""
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


"""
Model definition & Compiliation
"""
model = DenseNet121(weights=None, classes=10, input_shape=(32, 32, 3))
model.compile(optimizer, loss_object)

result = np.zeros(n_epochs)

"""
Training
"""
train_generator = datagen.flow(x_train, y_train, batch_size=x_train.shape[0])

for epoch in range(n_epochs):
    img, label = train_generator.next()
    model.fit(img, label, batch_size = batch_size)
    
    for images, labels in test_ds:
        test_step(images, labels, model)
        
    template = 'Epoch: {}, 테스트 손실: {}, 테스트 정확도: {}'
    print (template.format(
                         epoch,
                         test_loss.result(),
                         test_accuracy.result()*100))
    result[epoch] = test_loss.result()

    test_accuracy.reset_states()
    test_loss.reset_states()
    
    gc.collect()



"""
File Transfer
"""
filename = 'result.csv'

np.savetxt(filename, result)

ServerIp = '166.104.245.218'

# Now we can create socket object
s = socket.socket()

# Lets choose one port and connect to that port
PORT = 11821

# Lets connect to that port where server may be running
s.connect((ServerIp, PORT))

# We can send file sample.txt
file = open("result.csv", "rb")
SendData = file.read(1024)


while SendData:
    #Now send the content of sample.txt to server
    s.send(SendData)
    SendData = file.read(1024)      

file.close()
s.close()

print("Experiment Done.")
