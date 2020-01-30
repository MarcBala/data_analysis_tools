# data_analysis_tools
data_analysis_tools is a package to analyse data from Marc's experiment 

It allows to
+   load binary and csv files from Marc's acquisition setup.
+	fit experimenal data and extract physical quanties from the fit parameters
+	visualizes experimental data and theory utilizing the standard python [matplotlib](https://matplotlib.org/) library

data_analysis_tools is built on python 3.6 and tested on macOS  and Windows
It was built by Marc Torrent at ICFO. 
It is distributed under the [Revised BSD License](https://en.wikipedia.org/wiki/BSD_licenses).

## Getting Started
The software was developed and tested with python 3.6 on 64-bit Ubunutu. Prior to installation, install the latest  Anaconda distribution for python version 3.6, as it contains some extra dependencies this project utilizes.
You can find the latest Anaconda distribution [here](https://www.continuum.io/downloads). 

### Installation
There are two main ways to install data_analysis_tools: via pip, the python package manager, or directly from the source via github. The former is easier, while the latter gives more explicit access to the source code.

#### Via pip (Beginner)
The simplest way to install data_analysis_tools is with the command-line utility pip. To install simply issue the command

```>>> pip install git+https://github.com/MarcBala/data_analysis_tools.git```

#### Via git (Intermediate/Advanced)
If you are interested in hosting the source code more directly, you can clone from our git page:

```>>> git clone https://github.com/MarcBala/data_analysis_tools.git```

# License
This software is released under a dual license; one of the following options can be chosen:

The [Revised BSD License](https://opensource.org/licenses/BSD-2-Clause) (© 2020, Marc Torrent [MT]).
Any other license, as long as it is obtained from the creator of this package.

## FAQ
### pip install doesn't work
Make sure you have the latest version of pip and setuptools
```>>> pip install --upgrade setuptools```
```>>> pip install --upgrade pip```

