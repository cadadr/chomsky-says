# v01/chomsky-says.py --- initial version of the chomsky bot.

# This version is based on:
#   <https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/>
# and is capable of generating interesting strings, but no proper
# sentences.

import nltk
import numpy
import sys
import time

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

nltk.download('stopwords')

def tokenise(string):
    s = string.lower()
    t = RegexpTokenizer(r'\w+').tokenize(s)
    f = lambda w: w not in stopwords.words('english')
    return " ".join(filter(f, t))

t = open("../choms.txt").read()
p = tokenise(t[len(t)//2:len(t)//2+len(t)//2500])
c = sorted(list(set(p)))
n = dict((c, i) for i, c in enumerate(c))

print("Total number of characters:", len(p))
print("Total vocab:", len(c))

s = 100
x = []
y = []

for i in range(0, len(p) - s, 1):
    ins  = p[i:i+s]
    outs = p[i+s]
    x.append([n[c] for c in ins])
    y.append(n[outs])

print("Total patterns:", len(x))

X = numpy.reshape(x, (len(x), s, 1))
X = X / float(len(c))

Y = np_utils.to_categorical(y)

m = Sequential()
m.add(
    LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)
)
m.add(Dropout(0.2))
m.add(LSTM(256, return_sequences=True))
m.add(Dropout(0.2))
m.add(LSTM(128))
m.add(Dropout(0.2))
m.add(Dense(Y.shape[1], activation='softmax'))

m.compile(loss='categorical_crossentropy', optimizer='adam')

f = "model_weights_saved.hdf5"

cp = ModelCheckpoint(
    f, monitor='loss', verbose=1, save_best_only=True, mode='min'
)
cb = [cp]

m.fit(X, Y, epochs=16, batch_size=256, callbacks=cb)

m.load_weights(f)
m.compile(loss='categorical_crossentropy', optimizer='adam')

n = dict((i, c) for i, c in enumerate(c))

while True:
    s = numpy.random.randint(0, len(x) - 1)
    p = x[s]
    w = ''.join([n[v] for v in p])
    # first and last words tend to be junk
    w = " ".join(w.split()[1:-1])
    print("Chomsky says:", w)
    print("Chomsky is thinking now...")
    time.sleep(5)
