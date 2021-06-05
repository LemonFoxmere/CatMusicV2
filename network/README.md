## How To Train Your Own Custom Network With Custom Parameters and Stuff
---
**WARNING:** This process is not guaranteed to work, as the code is not fully reliable to work cross platformed in different environments.

Train and save the neural network with `TrainModel.py` found within `../source/network`

You can mess around with the code inside, but a fair warning to you is that you might get brain cancer trying to understand it

I labeled important training parameters in the code, so if you want you can tinker with those first.

After you finished tinkering (or not), you can simply run the training script (assuming all training and testing data are generated correctly), and watch the neural network learn, and your computer fan potentially scream.

If you want to run this script on the GPU, this is what worked for me:
* TensorFlow (normal and GPU version) == `2.5.0`
* Cuda version == `11.2`
* GPU Compute Cap == `8.6`
* Driver Version == `460.27.04`

If you can't get the GPU version running, don't worry about it, ~~it wasn't meant to be user frien...~~ uuh I mean it's still a prototype product, so there wasn't much thought put into this yet.
