<div align="center">
<br/>
<div align="left">
<br/>
<p align="center">
<a href="https://github.com/wilhelmagren/synd">
<img align="center" width=40% src="https://github.com/wilhelmagren/synd/blob/120ad15bf411807073b7f279c6390560ae1054c3/docs/images/synd-transparent.png"></img>
</a>
</p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lines of code](https://img.shields.io/tokei/lines/github/wilhelmagren/synd)](https://github.com/wilhelmagren/synd/tree/58bcd31b37c5bde0c8656717ed6c0f81cc3ec562/synd)

</div>

## 🔎 Overview
SYNthetic Data generation for complex tabular datasets.

## 🔒 Requirements
- At least Python3.6 
- requirements.txt
- for fast training and sampling you need CUDA compatible GPU

## 📦 Installation
Either clone this repository and perform a local install accordingly
```
git clone https://github.com/wilhelmagren/synd.git
cd synd
pip install -e .
```
or install the most recent release from the Python Package Index (PyPI).
```
pip install <tbd>
```

## 🚀 Example usage
```python
from sdv.datasets.demo import download_demo
data, metadata = download_demo('single_table', 'adult')

from synd import TGAN
from synd.datasets import SingleTable

discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'label',
]

dataset = SingleTable(data, metadata.to_dict(), discrete_columns=discrete_columns)
dataset.fit()

model = TGAN(batch_size=batch_size, device=device)
model.fit(dataset, epochs=epochs)

fake_samples = model.sample(n_samples=1200)

from sdmetrics.reports.single_table import QualityReport
report = QualityReport()
report.generate(data, fake_samples, metadata.to_dict())
...
```


## 📋 License
All code is to be held under a general MIT license, please see [LICENSE](https://github.com/wilhelmagren/synd/blob/fa06666402cfa0aa05846c9513aff19fc720a8f1/LICENSE) for specific information.
