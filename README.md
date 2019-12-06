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
conda create -n pnpbi -c pytorch python=3.6 matplotlib
```

Then, activate the environment:

```bash
conda activate pnpbi 
```

In order to run/edit the scripts using an IDE, install e.g. Spyder:

```bash
conda install -c pytorch spyder 
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

Image "Cat" by Lola Williams03 is licensed under CC0 1.0
URL: https://www.flickr.com/photos/161321817@N06/38633459455
