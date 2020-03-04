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
 - `test_fraction` is 0-1 fraction determining the size of the test set (set to 0 to disable testing),
 - and `optional arguments` depend on the chosen model, see the contents of the `models/` directory for details.


### Visualizing models

The package exposes the `tbr_visualizer` GUI endpoint to enable model inspection and visualization.
See [the implementation](./tbr_reg/visualizer.py) for details. Common usage is as follows:

```bash
tbr_visualizer
```

License
-------

This work was realised in 2020 as a group project at University College London with the support from UKAEA. The authors of the implementation are Petr Mánek and Graham Van Goffrier.

Permission to use, distribute and modify is hereby granted in accordance with the MIT License. See the LICENSE file for details.