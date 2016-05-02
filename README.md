# Neurphys

Neurphys (pronounced "nervous") is an IO and analysis package built to streamline and standardize the data handling, analysis, and visualization of electrophysiology and calcium imaging data.


## Dependencies

- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [pandas](http://pandas.pydata.org/)
- [neo](https://pythonhosted.org/neo/)
- [lxml](http://lxml.de/)


## Installation

First, download [Anaconda](https://www.continuum.io/downloads) which will come with
most required libraries including `numpy`, `pandas`, `scipy`.
Then download or clone the repository using `git` as follows,

```bash
$ git clone https://github.com/surmeierlab/neurphys
```

You can download dependencies which we provide in `requirements.txt`. Use `pip` to install
the rest of dependencies i.e.

```bash
$ pip install -r requirements.txt
```

Install `neurphys` using `setup.py` as follows

```bash
$ python setup.py develop install
```


## Usage

```python
import neurphys as nu
```

Check out the [tutorials page](https://github.com/surmeierlab/tutorials) for jupyter notebooks showing specific use cases.
