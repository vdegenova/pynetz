Pynetz
======


A Simple Python Neural Network Library
--------------------------------------

Have you ever wanted to have the ability to play around with a simple Neural Network in Python and train it to complete
tasks of your choosing? Pynetz allows you to work with Neural Networks written in Python and provides a clean and easy
interface to work with.


Installation
------------

From PyPi::

    pip install pynetz


Or from source::

    python setup.py install


Creating Your Own Neural Networks
---------------------------------

::

   import numpy as np
   # initialize a Neural Network with 3 inputs at the input, a hidden layer with 100 units, and an output layer with 3 units
   n = NN([3, 100, 3])
   vals = np.random.random((100, 3))
   X = np.split(vals, 100)  # splitting into multiple training examples
   Y = np.sin(x)  # training the network to learn sin(x) as a simple example
   eta = 0.1  # learning rate
   epochs = 50  # max number of epochs
   gen_plots = False  # don't generate plots for training error
   n.train(epochs, X, Y, eta, gen_plots)


Sampling Your Neural Network
----------------------------

::

   print "Expected: ", np.sin(np.array([0.8, 0.9, 0.2]))
   print "Sampled: ", n.feedforward(np.array([0.8, 0.9, 0.2]))