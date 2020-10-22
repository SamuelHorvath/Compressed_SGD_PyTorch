# Code guidelines

This implementation is based on PyTorch (1.5.0) in Python (3.8). 

It enables to run simulated distributed optimization with master node on any number of workers based on [PyTorch SGD Optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) with gradient compression. Communication can be compressed on both workers and master level. Error-Feedback is also enabled. For more details, please see our [manuscript](https://arxiv.org/pdf/2006.11077.pdf).

### Installation

To install requirements
```sh
$ pip install -r requirements.txt
```

###  Example Notebook
To run our code see [example notebook](example_notebook.ipynb).

### Citing
In case you find this this code useful, please consider citing

```
@article{horvath2020better,
  title={A Better Alternative to Error Feedback for Communication-Efficient Distributed Learning},
  author={Horv\'{a}th, Samuel and Richt\'{a}rik, Peter},
  journal={arXiv preprint arXiv:2006.11077},
  year={2020}
}
```

### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
