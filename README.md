# Astronomical Object Classifier

Classify astronomical objects. Dataset from https://www.kaggle.com/datasets/engeddy/astrophysical-objects-image-dataset.

## Training
Run `trainer.py`. See `python trainer.py --help` for all options. You can view tensorboard logs by running `tensorboard --logdir logs`. Furthermore, the best model based on validation loss will be saved under the `assets/` directory.

## Evaluation
Run `Evaluator.ipynb` to evaluate the models.
