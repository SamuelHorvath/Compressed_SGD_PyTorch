# Code guidelines

This implementation is based on PyTorch (1.5.0) in Python (3.8). 

It enables to run simulated distributed optimization with master node on any number of workers based on [PyTorch SGD Optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) with gradient compression. Communication can be compressed on both on workers and master level. Error-Feedback is also enabled. For more details, please see our [manuscript](https://arxiv.org/pdf/2006.TBD.pdf)

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
  author={Horv{\'a}th, Samuel and Ho, and Richt{\'a}rik, Peter},
  journal={arXiv preprint arXiv:2006.TBD},
  year={2020}
}
```

### Contact
In case of any question, please contact [Samuel Horvath](mailto:samohorvath11@gmail.com).

### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
