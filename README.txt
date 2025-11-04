SICGAN - Single Image Colourisation using a Generative Adversarial Network.

Read SICGAN.pdf to understand the goals, results, and foundations of this project. This was a university project, it should be noted though that due to imposed page limits for the submission the report is not as extensive as I would like and it does not clearly state that the training and test data sets ARE different. For training data, patches are randomly sampled and augmented, whereas for testing data, patches are linearly sampled along the image. This means that this deep learning task is a generative task and NOT a memorisation task.

This was written with python 3.10.

To run this code there are a couple options. If you want to run the program locally, it will detect if there is a cuda compatible GPU (to run the cuda version of pytorch for GPU acceleration), otherwise it will run on the CPU (NOT RECOMMENDED THIS WILL MEAN YOU ARE SITTING THERE FOR A VERY VERY LONG TIME)

However, if you do use a cpu, please tweak the number of patches to a suitably low number ~500-1000 and if it still trains very slowly reduce all mentions of the patch size to 64x64. THIS WILL IMPACT THE RESULTS BUT WILL SPEED UP TRAINING.

NOTE that you can't have the normal version of torch, the cuda enabled version is required if you want to use a GPU. The required libraries are all listed in requirements.txt. This can be installed using "pip install -r requirements.txt" or you can install these all manually.

The easier solution is just to connect to google colab and connect to a GPU there.

Running the code is as simple as "python sicgan.py" ensuring that "jcsmr.jpg" is in the same directory as the program.

