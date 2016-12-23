import numpy as np
import numpy as np
from random import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

# Prepare training data.
print("Preparing Data.")
from lib import vectorize
from lib import text_searcher

vec = vectorize.Vectorizer("char1.txt", "label.txt")
searcher = text_searcher.TextSearcher("corpus.sqlite")

CHAR_SPACE = 0
TEXT_LENGTH = 100
    #test data %
test_percent = 0.1

trainingFile = open("training_terms.tsv", "r", encoding="utf-8").read().strip().split("\n")

training_terms = []
for row in trainingFile:
    r = row.split("\t")
    x = r[0]
    y = r[1].split(" ")
    training_terms.append((x, y))


training_data = []
for terms in training_terms:
    query = terms[0]
    n = 0
    searcher = text_searcher.TextSearcher("corpus.sqlite")
    for doc in searcher.genDocs(query):
        n += 1
        if n >= 10:
            break
        training_data.append((doc, terms[1]))


X_samples = []
y_samples = []
for data in training_data:
    x = vec.vectorize(data[0], TEXT_LENGTH)
    y = vec.vectorizeLabel(data[1])
    X_samples.append(x)
    y_samples.append(y)

shuffle(X_samples)
nb_test = int(len(X_samples) * test_percent)

X_train = np.array(X_samples[nb_test:])
y_train = np.array(y_samples[nb_test:])
X_test = np.array(X_samples[:nb_test])
y_test = np.array(y_samples[:nb_test])

# For Convolution2D input. 4D with shape (samples, rows, cols, channels)
CHAR_SPACE = X_train.shape[1]

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, CHAR_SPACE, TEXT_LENGTH)
    X_test = X_test.reshape(X_test.shape[0], 1, CHAR_SPACE, TEXT_LENGTH)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], CHAR_SPACE, TEXT_LENGTH, 1)
    X_test = X_test.reshape(X_test.shape[0], CHAR_SPACE, TEXT_LENGTH, 1)
    input_shape = (CHAR_SPACE, TEXT_LENGTH, 1)


print("Loaded training data: {}".format(len(training_terms)))
print("Convert to docs: {}".format(len(training_data)))
print("Training samples: {}".format(X_train.shape[0]))
print("Test samples: {}".format(X_test.shape[0]))
print("Array shape: x:{}, y:{}".format(X_samples[0].shape, y_samples[0].shape))

# Model hyperparameter setting.
NB_FILTER = 32
NB_GRAM = 2

input_shape = X_samples[0].shape + (1,)
output_shape = y_samples[0].shape
nb_char = input_shape[0]
nb_classes = y_samples[0].shape[0]

epochs = 4
batch_size = 32

# Create the model
model = Sequential()
model.add(Convolution2D(NB_FILTER, nb_char, NB_GRAM, input_shape=input_shape, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 99)))
model.add(Flatten())
model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
print(model.summary())

# Fit the model
print("Start fitting model.")
model.fit(X_train, y_train, validation_split=0.15, nb_epoch=epochs, batch_size=batch_size)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
