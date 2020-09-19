# Code2vec
A neural network for learning distributed representations of code.
This is the modified implementation of the model described in:

**Learning Off-By-One Mistakes: An Empirical Study on Different Deep Learning Models**  

## See also:
  * **code2seq** (ICLR'2019) is our newer model. It uses LSTMs to encode paths node-by-node (rather than monolithic path embeddings as in code2vec), and an LSTM to decode a target sequence (rather than predicting a single label at a time as in code2vec). See [PDF](https://openreview.net/pdf?id=H1gKYo09tX), demo at [http://www.code2seq.org](http://www.code2seq.org) and [code](https://github.com/tech-srl/code2seq/).

This is a TensorFlow implementation, designed to be easy and useful in research, 
and for experimenting with new ideas in machine learning for code tasks.
By default, it learns Java source code and predicts Java method names, but it can be easily extended to other languages, 
since the TensorFlow network is agnostic to the input programming language (see [Extending to other languages](#extending-to-other-languages).
Contributions are welcome.
This repo actually contains two model implementations. The 1st uses pure TensorFlow and the 2nd uses TensorFlow's Keras ([more details](#choosing-implementation-to-use)). 

<center style="padding: 40px"><img width="70%" src="https://github.com/tech-srl/code2vec/raw/master/images/network.png" /></center>

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Configuration](#configuration)
  * [Features](#features)
  * [Extending to other languages](#extending-to-other-languages)
  * [Additional datasets](#additional-datasets)
  * [Citation](#citation)

## Requirements
#### On Ubuntu:
  * [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04) (>=3.6). To check the version:
> python3 --version
  * Create virtual environment and install packages (Tensorflow & Numpy)
> python3 -m pip install virtualenv --user  
> python3 -m venv env  
> source env/bin/activate  
> pip install -r requirements.txt  
  * TensorFlow - version 2.0.0 ([install](https://www.tensorflow.org/install/install_linux)).
  To check TensorFlow version:
#### Use Virtualenv  
  * Run following commands to get python requirements installed and virtual environment activated
> python3 -m pip install virtualenv  
> source virtualenv.sh
#### Or install pip packages manually
> python3 -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'
  * `pip install wandb` [wandb](https://www.wandb.com/) for logging metrics during training  
  * `pip install pyyaml` it is required for wandb and does not get installed automatically sometimes  
#### CUDA
* If you are using a GPU, you will need CUDA 10.0
  ([download](https://developer.nvidia.com/cuda-10.0-download-archive-base)) 
  as this is the version that is currently supported by TensorFlow. To check CUDA version:
> nvcc --version
  * For GPU: cuDNN (>=7.5) ([download](http://developer.nvidia.com/cudnn)) To check cuDNN version:
> cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
  * For [creating a new dataset](#creating-and-preprocessing-a-new-java-dataset)
  or [manually examining a trained model](#step-4-manual-examination-of-a-trained-model)
  (any operation that requires parsing of a new code example) - [Java JDK](https://openjdk.java.net/install/)

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/hsellik/thesis
cd code2vec
```

### Step 1: Creating a new dataset from java sources
In order to have a preprocessed dataset to train a network on, you can create a new dataset of your own.

#### Creating and preprocessing a new Java dataset
In order to create and preprocess a new dataset (for example, to compare code2vec to another model on another dataset):
  * Edit the file [preprocess.sh](preprocess.sh) using the instructions there, pointing it to the correct training, validation and test directories.
  * Run the preprocess.sh file:
> source preprocess.sh

### Step 2: Training a model
You can either download an already-trained model, or train a new model using a preprocessed dataset.

#### Training a model from scratch
To train a model from scratch:
  * Edit the file [train.sh](train.sh) to point it to the right preprocessed data. By default, 
  it points to our "java14m" dataset that was preprocessed in the previous step.
  * Before training, you can edit the configuration hyper-parameters in the file [config.py](config.py),
  as explained in [Configuration](#configuration).
  * Run the [train.sh](train.sh) script:
```
source train.sh
```

##### Notes:
  1. By default, the network is evaluated on the validation set after every training epoch.
  2. The newest 10 versions are kept (older are deleted automatically). This can be changed, but will be more space consuming.
  3. By default, the network is training for 20 epochs.
These settings can be changed by simply editing the file [config.py](config.py).
Training on a Tesla v100 GPU takes about 50 minutes per epoch. 
Training on Tesla K80 takes about 4 hours per epoch.

### Step 3: Evaluating a trained model
Once the score on the validation set stops improving over time, you can stop the training process (by killing it)
and pick the iteration that performed the best on the validation set.
Suppose that iteration #8 is our chosen model, run:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8.release --test data/java14m/java14m.test.c2v
```
While evaluating, a file named "log.txt" is written with each test example name and the model's prediction.

### Step 4: Manual examination of a trained model
To manually examine a trained model, run:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8.release --predict
```
After the model loads, follow the instructions and edit the file [Input.java](Input.java) and enter a Java 
method or code snippet, and examine the model's predictions and attention scores.

## Configuration
Changing hyper-parameters is possible by editing the file
[config.py](config.py).

Here are some of the parameters and their description:
#### config.NUM_TRAIN_EPOCHS = 20
The max number of epochs to train the model. Stopping earlier must be done manually (kill).
#### config.SAVE_EVERY_EPOCHS = 1
After how many training iterations a model should be saved.
#### config.TRAIN_BATCH_SIZE = 1024 
Batch size in training.
#### config.TEST_BATCH_SIZE = config.TRAIN_BATCH_SIZE
Batch size in evaluating. Affects only the evaluation speed and memory consumption, does not affect the results.
#### config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION = 10
Number of words with highest scores in $ y_hat $ to consider during prediction and evaluation.
#### config.NUM_BATCHES_TO_LOG_PROGRESS = 100
Number of batches (during training / evaluating) to complete between two progress-logging records.
#### config.NUM_TRAIN_BATCHES_TO_EVALUATE = 100
Number of training batches to complete between model evaluations on the test set.
#### config.READER_NUM_PARALLEL_BATCHES = 4
The number of threads enqueuing examples to the reader queue.
#### config.SHUFFLE_BUFFER_SIZE = 10000
Size of buffer in reader to shuffle example within during training.
Bigger buffer allows better randomness, but requires more amount of memory and may harm training throughput.
#### config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
The buffer size (in bytes) of the CSV dataset reader.

#### config.MAX_CONTEXTS = 200
The number of contexts to use in each example.
#### config.MAX_TOKEN_VOCAB_SIZE = 1301136
The max size of the token vocabulary.
#### config.MAX_TARGET_VOCAB_SIZE = 261245
The max size of the target words vocabulary.
#### config.MAX_PATH_VOCAB_SIZE = 911417
The max size of the path vocabulary.
#### config.DEFAULT_EMBEDDINGS_SIZE = 128
Default embedding size to be used for token and path if not specified otherwise.
#### config.TOKEN_EMBEDDINGS_SIZE = config.EMBEDDINGS_SIZE
Embedding size for tokens.
#### config.PATH_EMBEDDINGS_SIZE = config.EMBEDDINGS_SIZE
Embedding size for paths.
#### config.CODE_VECTOR_SIZE = config.PATH_EMBEDDINGS_SIZE + 2 * config.TOKEN_EMBEDDINGS_SIZE
Size of code vectors.
#### config.TARGET_EMBEDDINGS_SIZE = config.CODE_VECTOR_SIZE
Embedding size for target words.
#### config.MAX_TO_KEEP = 10
Keep this number of newest trained versions during training.
#### config.DROPOUT_KEEP_RATE = 0.75
Dropout rate used during training.
#### config.SEPARATE_OOV_AND_PAD = False
Whether to treat `<OOV>` and `<PAD>` as two different special tokens whenever possible.

## Features
Code2vec supports the following features: 

### Choosing implementation to use
This repo comes with two model implementations:
(i) uses pure TensorFlow (written in [tensorflow_model.py](tensorflow_model.py));
(ii) uses TensorFlow's Keras (written in [keras_model.py](keras_model.py)).
The default implementation used by `code2vec.py` is the pure TensorFlow.
To explicitly choose the desired implementation to use, specify `--framework tensorflow` or `--framework keras`
as an additional argument when executing the script `code2vec.py`.
Particularly, this argument can be added to each one of the usage examples (of `code2vec.py`) detailed in this file.
Note that in order to load a trained model (from file), one should use the same implementation used during its training.

### Releasing the model
If you wish to keep a trained model for inference only (without the ability to continue training it) you can
release the model using:
```
python3 code2vec.py --load models/java14_model/saved_model_iter8 --release
```
This will save a copy of the trained model with the '.release' suffix.
A "released" model usually takes 3x less disk space.

## Extending to other languages  

This project currently supports Java and C\# as the input languages.

_**January 2020** - an extractor for predicting TypeScript type annotations for JavaScript input using code2vec was developed by [@izosak](https://github.com/izosak) and Noa Cohen, and is available here:
[https://github.com/tech-srl/id2vec](https://github.com/tech-srl/id2vec)._

~~_**June 2019** - an extractor for **C** that is compatible with our model was developed by [CMU SEI team](https://github.com/cmu-sei/code2vec-c)._~~ - removed by CMU SEI team.

_**June 2019** - an extractor for **Python, Java, C, C++** by JetBrains Research is available here: [PathMiner](https://github.com/JetBrains-Research/astminer)._

In order to extend code2vec to work with other languages, a new extractor (similar to the [JavaExtractor](JavaExtractor))
should be implemented, and be called by [preprocess.sh](preprocess.sh).
Basically, an extractor should be able to output for each directory containing source files:
  * A single text file, where each row is an example.
  * Each example is a space-delimited list of fields, where:
  1. The first "word" is the target label, internally delimited by the "|" character.
  2. Each of the following words are contexts, where each context has three components separated by commas (","). Each of these components cannot include spaces nor commas.
  We refer to these three components as a token, a path, and another token, but in general other types of ternary contexts can be considered.  

For example, a possible novel Java context extraction for the following code example:
```java
void fooBar() {
	System.out.println("Hello World");
}
```
Might be (in a new context extraction algorithm, which is different than ours since it doesn't use paths in the AST):
> foo|Bar System,FIELD_ACCESS,out System.out,FIELD_ACCESS,println THE_METHOD,returns,void THE_METHOD,prints,"hello_world" 

Consider the first example context "System,FIELD_ACCESS,out". 
In the current implementation, the 1st ("System") and 3rd ("out") components of a context are taken from the same "tokens" vocabulary, 
and the 2nd component ("FIELD_ACCESS") is taken from a separate "paths" vocabulary. 

## Additional datasets
We preprocessed additional three datasets used by the [code2seq](https://arxiv.org/pdf/1808.01400) paper, using the code2vec preprocessing.
These datasets are available in raw format (i.e., .java files) at [https://github.com/tech-srl/code2seq/blob/master/README.md#datasets](https://github.com/tech-srl/code2seq/blob/master/README.md#datasets),
and are also available to download in a preprocessed format (i.e., ready to train a code2vec model on) here:

### Java-small (compressed: 366MB, extracted 1.9GB)
```
wget https://s3.amazonaws.com/code2vec/data/java-small_data.tar.gz
```
This dataset is based on the dataset of [Allamanis et al. (ICML'2016)](http://groups.inf.ed.ac.uk/cup/codeattention/), with the difference that training/validation/test are split by-project rather than by-file.
This dataset contains 9 Java projects for training, 1 for validation and 1 testing. Overall, it contains about 700K examples.

### Java-med (compressed: 1.8GB, extracted 9.3GB)
```
wget https://s3.amazonaws.com/code2vec/data/java-med_data.tar.gz
```
A dataset of the 1000 top-starred Java projects from GitHub. It contains
800 projects for training, 100 for validation and 100 for testing. Overall, it contains about 4M examples.

### Java-large (compressed: 7.2GB, extracted 37GB)
```
wget https://s3.amazonaws.com/code2vec/data/java-large_data.tar.gz
```
A dataset of the 9500 top-starred Java projects from GitHub that were created
since January 2007. It contains 9000 projects for training, 200 for validation and 300 for
testing. Overall, it contains about 16M examples.

## Citation

[code2vec: Learning Distributed Representations of Code](https://urialon.cswp.cs.technion.ac.il/wp-content/uploads/sites/83/2018/12/code2vec-popl19.pdf)

```
@article{alon2019code2vec,
 author = {Alon, Uri and Zilberstein, Meital and Levy, Omer and Yahav, Eran},
 title = {Code2Vec: Learning Distributed Representations of Code},
 journal = {Proc. ACM Program. Lang.},
 issue_date = {January 2019},
 volume = {3},
 number = {POPL},
 month = jan,
 year = {2019},
 issn = {2475-1421},
 pages = {40:1--40:29},
 articleno = {40},
 numpages = {29},
 url = {http://doi.acm.org/10.1145/3290353},
 doi = {10.1145/3290353},
 acmid = {3290353},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Big Code, Distributed Representations, Machine Learning},
}
```
