import tensorflow as tf
import numpy as np

output = np.array([0,0,0,0,0, # The output we actually want
                   0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0,
                   1,0,1,1,0,# <- These 3 ones are suppose to represent
                   0,0,0,0,0,#    the activation of those specific notes
                   0,0,0,0,0], dtype=np.float32)

input  = np.array([0,0,0,0,0, # The AI's bad generation
                   0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0,# <- in the AI's case it just completly
                   0,0,0,0,0,#    skips over it
                   0,0,0,0,0], dtype=np.float32)

# The losses that results:
tf.losses.binary_crossentropy(output, input).numpy() # The loss I am using gave:
tf.losses.mean_squared_error(output, input).numpy() #  The loss I was before using gave:
# The AI was able to exploit losses to minimize them while not doing what we want it to do.
