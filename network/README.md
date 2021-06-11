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

---

### Q&A
#### How does the naming convention work?
All the names are generated algorithmic, with the following naming convention: `snp_{batch_cycle}.h5`.

The bigger the batch cycle number is, the more the network has been trained. by default, snapshots are saved every 100 batch cycles, as well as the training and validation loss, which will be read in later by `Post.py`

The completed network (assuming that training was not interrupted), will be named `snp_fin.h5`. Usually if you synthesized the data right and trained the network good, this snapshot should yield the best result.

#### How does this weird output folder structure work?
by default, `TrainModel.py` will create a folder named `latest-cyc` (or if it already exist it'll just use that), and store everything in there. After each Training session, say you want to train again but with tuned parameters, you should rename the `latest-cyc` folder to something more appropriate.

#### How do I use `Post.py` to check the network's performance?
You have the option to run the file through a Jupyter notebook, but what I like to do is to just run it line by line in Atom with the Hydrogen plugin. Just don't forget to change your training session name as well as network snapshot name.
