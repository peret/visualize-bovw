visualize-BoVW
==============

These are the scripts I used for my master's thesis on *Visualizing Bag-of-Visual-Words for Visual Concept Detection*.
The code should only depend on [scikit-learn](https://github.com/scikit-learn/scikit-learn) (>= 0.14) and the
[Python Imaging Library](http://www.pythonware.com/products/pil/) (for the visualizations).

# Training classifiers and grid search

## DataManagers
Data managers provide an interface to work with different datasets. Their main purpose is to expose methods to
- get the input data matrix for a specific category
- get the vector of class labels for a specific category

for either a subset of the data or the whole dataset.

To implement a DataManager for a new dataset, you will have to inherit from `DataManager` and implement
three methods `_build_sample_matrix`, `_build_class_vector` and `_get_positive_samples`.
Two datamanagers for the ImageCLEF 2011 and Caltech101 dataset are included in this repository.
