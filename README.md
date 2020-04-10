# Image reconstruction using Bregman iteration and plug-and-play denoisers

## Dependencies

This software was written for and tested with:
- MacOS Mojave (version 10.14.6)
- Anaconda (conda version 4.7.12)
- Python (version 3.6.7)

The following libraries are required for parts of this code:

- pytorch
- matplotlib

## Installation

Download and install [Anaconda](https://anaconda.org/).

There are two ways to create the conda environment with the correct library versions:

a) Use the provided [`environment.yml`](environment.yml) file:

```bash
conda env create -f environment.yml
```

b) Manually create the environment:

```bash
conda create -n pnpbi -c astra-toolbox python=3.6 matplotlib astra-toolbox=1.8.3 scipy pillow=6.2.1 tqdm
```

Then, activate the environment:

```bash
conda activate pnpbi
```

Install PyTorch and torchvision:

```bash
conda install -c pytorch pytorch=1.3.1 torchvision=0.4.2
```

Install pydicom:

```bash
conda install -c conda-forge pydicom
```

In order to run/edit the scripts using an IDE, install e.g. Spyder:

```bash
conda install spyder
```

Alternatively, you can use e.g. PyCharm and create a run environment by selecting anaconda3/envs/pnpbi environment.

## Usage

To run the test cases, execute

```bash
python -m unittest discover
```

## License & Disclaimer

This code is released under GNU GPL version 3.
For the full license statement see the file [LICENSE](LICENSE).

## Contact

[Lukas F. Lang](https://lukaslang.github.io)
Department of Applied Mathematics and Theoretical Physics
University of Cambridge
Wilberforce Road, Cambridge CB3 0WA, United Kingdom


## Acknowledgements

Image ["Cat"](https://www.flickr.com/photos/161321817@N06/38633459455) by Lola Williams03 is licensed under CC0 1.0

Image "Brain" borrowed from:

```
M. Guerquin-Kern, L. Lejeune, K. P. Pruessmann, and M. Unser. Realistic analytical phantoms for parallel magnetic resonance imaging. IEEE Trans. Med. Imag., 31(3):626–636, 2012.
```
