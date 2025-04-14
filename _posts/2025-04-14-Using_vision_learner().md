# Using vision_learner() in fastai

During my exploration of the fastai course, one of the functions that really stood out to me was vision_learner(). It encapsulates the power of transfer learning in a single line and makes loading and training state-of-the-art computer vision models incredibly simple.

In this blog post, I’ll share how I learned to use vision_learner() to load pretrained models, and how fine_tune() leverages transfer learning to quickly achieve high accuracy — even with limited data.

## What is vision_learner()?

vision_learner() is a function in fastai that simplifies the process of creating a model for image classification. It takes care of:
- Loading a pretrained model architecture (like ResNet, EfficientNet, etc.),
- Attaching a new head (output layer) suited for your dataset,
- Creating a Learner object that can be used to train, evaluate, and export the model.

Here’s a quick example I worked with from the 00-is-it-a-bird Jupyter notebook:
```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
```
That’s it! With just one line using vision_learner(), I had a model loaded with pretrained weights from ResNet18, and it was ready to be fine-tuned.

## What is fine_tune()?

fine_tune() is another powerful fastai method that builds on transfer learning. When you call fine_tune() on a learner, it:
- Trains the newly added head (classification layers) for a few epochs while keeping the pretrained layers frozen.
- Unfreezes all the layers and continues training the entire network for a few more epochs.
This two-stage training process helps the model adapt better to your dataset without forgetting what it learned from the original dataset.

Here’s how I used it:
```python
learn.fine_tune(3)
```
And that’s it! In just a couple of minutes, I had a high-performing image classifier that could distinguish between birds and non-birds.

## Why this matters

Using vision_learner() and fine_tune() made me appreciate how accessible deep learning has become. Without writing chunks of code, I was able to:
- Load a sophisticated CNN model
- Automatically adapt it to my custom dataset
- Train it with minimal manual tuning
- Monitor progress with built-in metrics like accuracy

All this while using highly readable, intuitive code.

## Bonus: Customising Your Model

You can also customise your model easily. For instance, you can change the architecture (say from resnet18 to resnet34) like so:
```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
```
You can also add more metrics or change the loss function depending on your problem.

## Final thoughts

The vision_learner() and fine_tune() combo is one of the most powerful features in fastai. It abstracts away all the messy parts of setting up a deep learning model, letting you focus on learning and experimentation.

Whether you're building a classifier for pets, flowers, or even medical scans, these tools help you go from dataset to model in minutes — not hours.
