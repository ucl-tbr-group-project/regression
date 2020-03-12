TBR Regression
==============

This repository contains implementation of various regression approaches to approximate TBR without having to run TBR simulation. It is assumed that a sufficiently large data set of sampled TBR is provided.


Usage
-----

The repository provides the `tbr_reg` Python package. Use `pip` to install it on your system.

### Training & evaluating models

The package exposes the `tbr_train` command line endpoint to enable quick model training and evaluation.
See [the implementation](./tbr_reg/run_training.py) for details. Common usage is as follows:

```bash
tbr_train <model> <data_dir> <batch_start> <batch_end> <test_fraction> <optional arguments...>
```

where:

 - `model` is one of the supported models, see `model_loader.py` for details,
 - `data_dir` is path to directory containing CSV batch files,
 - `batch_start` and `batch_end` is range of batch file indices to use,
 - `test_fraction` can be either:
    - floating-point in the interval (0;1] determining the fractional size of the test set,
    - 0 to disable testing (in that case entire input is used to train the model),
    - a negative integer determining number of folds for cross-validation (e.g. `-5` yields 5-fold c.v.)
 - and `optional arguments` depend on the chosen model, see the contents of the [models/](./tbr_reg/models) directory for details.


### Visualizing models

The package exposes the `tbr_visualizer` GUI endpoint to enable model inspection and visualization.
See [the implementation](./tbr_reg/visualizer.py) for details. Common usage is as follows:

```bash
tbr_visualizer
```

### Evaluating models

The package exposes the `tbr_eval` command line endpoint to enable visual model evaluation.
See [the implementation](./tbr_reg/run_evaluation.py) for details. Common usage is as follows:

```bash
tbr_eval <data_dir> <batch_start> <batch_end> <model_file>
```

where:

 - `data_dir` is path to directory containing CSV batch files,
 - `batch_start` and `batch_end` is range of batch file indices to use,
 - `model_file` is a path to previously saved model file.

### Compressing dimensions with autoencoders

The package exposes the `tbr_ae` command line endpoint to enable lossy data compression with autoencoders.
See [the implementation](./tbr_reg/run_autoencoder.py) for details. Common usage is as follows:

```bash
tbr_ae <data_dir> <batch_start> <batch_end> <optional arguments...>
```


### Fixing discrete parameters

The package exposes the `tbr_split` command line endpoint to separate mixed data sets into groups selected
by fixing discrete parameters to constant values. See [the implementation](./tbr_reg/run_split_batches.py)
for details. Common usage is as follows:

```bash
tbr_split <data_dir> <output_dir> <batch_start> <batch_end> <optional arguments...>
```


License
-------

This work was realised in 2020 as a group project at University College London with the support from UKAEA. The authors of the implementation are Petr Mánek and Graham Van Goffrier.

Permission to use, distribute and modify is hereby granted in accordance with the MIT License. See the LICENSE file for details.