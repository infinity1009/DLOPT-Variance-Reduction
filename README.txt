# Implementation of SGD, SVRG and SAGA Optimizers

This repository provides Python implementations of three optimization algorithms: SGD, SVRG, and SAGA. These algorithms are derived from PyTorch's Optimizer class (located in `optimizers.py`). 

The implementations are evaluated on the FashionMNIST dataset, utilizing a convolutional neural network (defined in `network.py`).

## Optimizer Details

The optimizers are tested under various settings. While SGD and SVRG support mini-batch training, SAGA is currently configured to operate with `batch_size=1`. 

To ensure a fair comparison, SGD and SVRG are also experimented with `batch_size=1`.

## Usage

To execute the code, the following examples demonstrate the command-line usage:

- **SGD**:
  ```
  python run.py --epochs 300 --optimizer_type SGD --lr 0.005
  ```

- **SVRG**:
  ```
  python run.py --epochs 300 --optimizer_type SVRG --lr 0.005
  ```

- **SAGA**:
  ```
  python run.py --epochs 100 --optimizer_type SAGA --lr 0.005 --simple_model
  ```

Note: Due to SAGA's substantial memory consumption for auxiliary information, the `simple_model` flag must be set for testing. 

Refer to the argument settings in `run.py` for additional usage details.