## Project Structure and Description

This project is based on the [RETFound_MAE repository](https://github.com/uw-biomedical-ml/RETFound_MAE), with several optimizations. The structure and main components are described below:

- **`datasets/`**  
  Contains the dataset CSV files:
  - `train.csv`
  - `val.csv`
  - `test.csv`  
  Each file includes:
  - `image`: the path to the image
  - `labels`: the associated image labels

- **`util/`**  
  Includes various utility scripts and helper functions used throughout the project.

- **`engine_finetune_top1.py`**  
  Contains detailed training logic and procedures.

- **`main_finetune.py`**  
  The main script to run the training process. You can modify hyperparameters such as:
  - Dataset paths
  - Number of training epochs
  - Other training configurations

- **`models_vit.py`**  
  Defines the model architecture used in this project.

For more details, please refer to the original RETFound_MAE repository.
