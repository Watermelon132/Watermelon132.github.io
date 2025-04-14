# Learning about DataBlocks in fastai — and Customizing Batch Sizes
As part of my journey through the fastai course, I encountered one of the most powerful and flexible tools the library offers: the DataBlock API. 
Coming from a background in PyTorch, where data preprocessing often involves creating custom Dataset and DataLoader classes, I found fastai’s abstraction both elegant and beginner-friendly.

In this post, I’ll walk through what I learned about the DataBlock API, how it helped simplify data handling in deep learning workflows, and how I learned to customise batch sizes—a small but essential tweak that can significantly impact training performance.


## What is a DataBlock?

A DataBlock in fastai is essentially a blueprint for building a DataLoaders object. It defines a step-by-step pipeline for:

- Getting the input items (e.g., image files),
- Labeling the data (e.g., from filenames or folders),
- Applying transformations (e.g., resizing, normalization),
- Splitting the data into training and validation sets,
- Batching and feeding the data into the model.

Here’s a simple example I worked with from the 00-is-it-a-bird Jupyter notebook:
```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)
```
Once the DataBlock is defined, it’s easy to generate DataLoaders.
What really surprised me was how much could be packed into a single, declarative DataBlock line without sacrificing clarity.

## Changing the Batch Size

While experimenting, I noticed that training could sometimes be slow or cause memory issues on my GPU. That’s when I realized I could change the batch size (i.e. how many samples are processed at once during training).

In fastai, adjusting the batch size is very straightforward — just pass the bs parameter when calling .dataloaders():
```python
dls = {}.dataloaders(path, bs=64)
```
This changed the batch size from the default (usually 64) to a new value of my choice. On a smaller GPU, reducing bs can help prevent out-of-memory errors, while increasing it (if memory allows) can speed up training.

I experimented with various batch sizes (e.g., 16, 32, 64, 128) and observed the trade-offs in memory consumption and training time.

## Final thoughts

Learning about DataBlock in fastai has been a game-changer for me. Additionally, tweaking the batch size gave me greater control over resource usage during training.
If you’re just starting with fastai, I’d highly recommend diving into the DataBlock API — it’s an intuitive way to learn how data flows into your models.
