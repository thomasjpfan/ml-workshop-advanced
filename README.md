# Advanced Machine Learning with scikit-learn
### Text Data, Imbalanced Data, and Poisson Regression

*By Thomas J. Fan*

[Link to slides](https://thomasjpfan.github.io/ml-workshop-advanced/)

Scikit-learn is a machine learning library in Python that is used by many data science practitioners. In this training, we will learn about processing text data, working with imbalanced data and Poisson regression. We will start by learning about processing text data with scikit-learn's CountVectorizer and TfidfVectorizer. Since the output of these vectorizers are sparse matrices, we will also review the scikit-learn estimators that can handle sparse input data. Next, we will learn about how to work with imbalanced data which appears in datasets where one of the classes appears more frequently than the others. Next, we will learn about generalized linear models with a focus on Poisson regression. Poisson regression is used to model target distributions that are counts or relative frequencies.  Lastly, we will learn how to use tree-based models such as Histogram-based Gradient Boosted Trees with a poisson loss to model relative frequencies.

## Obtaining the Material

### With git

The most convenient way to download the material is with git:

```bash
git clone https://github.com/thomasjpfan/ml-workshop-advanced
```

Please note that I may add and improve the material until shortly before the session. You can update your copy by running:

```bash
git pull origin master
```

### Download zip

If you are not familiar with git, you can download this repository as a zip file at: [github.com/thomasjpfan/ml-workshop-advanced/archive/master.zip](https://github.com/thomasjpfan/ml-workshop-advanced/archive/master.zip). Please note that I may add and improve the material until shortly before the session. To update your copy please re-download the material a day before the session.

## Installation

This workshop requires conda to be installed on your local machine. The easiest install conda is to install `miniconda` by using an installer for your operating system provided by [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). After conda is installed, navigate to this repository on your local machine:

```bash
cd ml-workshop-advanced
```

To download and install the dependencies run:

```bash
conda env create -f environment.yml
```

This will install a virtual environment. To activate the environment:

```bash
conda activate ml-workshop-advanced
```

Finally, to start `jupyterlab` run:

```bash
jupyter lab
```

This should open a browser window with the `jupterlab` interface.

## License

This repo is under the [MIT License](LICENSE).
