# Astronomical Object Classifier

Classify astronomical objects. Dataset from https://www.kaggle.com/datasets/engeddy/astrophysical-objects-image-dataset.

## Setup

This project uses Python 3.10. 

### Initial Setup

1. **Set up the environment:**
   ```bash
   ./setup.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Download the data:**
   ```bash
   python download.py
   ```
   This will download data which you can move into the `data/` directory (which is gitignored).

## Training
Run `trainer.py`. See `python trainer.py --help` for all options. You can view tensorboard logs by running `tensorboard --logdir logs`. Furthermore, the best model based on validation loss will be saved under the `assets/` directory.

## Evaluation
Run `Evaluator.ipynb` to evaluate the models.
