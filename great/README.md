# Global Relational Models of Source Code
This is the modified implementation of the model described in:

[ICLR 2020 paper](http://vhellendoorn.github.io/PDF/iclr2020.pdf) on models of source code that combine global and structural information, including the Graph-Sandwich model family and the GREAT (Graph-Relational Embedding Attention Transformer) model.

Used for Master Thesis: **Learning Off-By-One Mistakes: An Empirical Study on Different Deep Learning Models**  

## Quick Start
The modeling code is written in Python (3.6+) and uses Tensorflow (recommended 2.2.x+). For a quick setup, run `pip install -r requirements.txt`.

To run training, first create datasets with [J2Graph](https://github.com/SERG-Delft/j2graph), which will create vocab builder and note its location (lets call it `*data_dir*`). Then run vocabulary_builder.py to generate a bpe-vocab.txt. Then, from the main directory of this repository, run: `python running/run_model.py *data_dir* vocab.txt config.yml`, to train the model configuration specified in `config.yml`, periodically writing checkpoints (to `models/` and evaluation results (to `log.txt`). Both output paths can be optionally set with `-m` and `-l` respectively.

To customize the model configuration, you can change both the hyper-parameters for the various model types available (transformer, GREAT, GGNN, RNN) in `config.yml`, and the overall model architecture itself under `model: configuration`. For instance, to train the RNN Sandwich architecture from our paper, set the RNN and GGNN layers to reasonable values (e.g. RNN to  2 layers and the GGNN's `time_steps` to \[3, 1\] layers as in the paper) and specify the model configuration: `rnn ggnn rnn ggnn rnn`.

To evaluate the trained model with the highest heldout accuracy, run: `python running/run_model.py *data_dir* vocab.txt config.yml -m *model_path* -l *log_path* -e True` (`model_path` and `log_path` are mandatory in this setting). This will run an evaluation pass on the entire 'eval' portion of the dataset and print the final losses and accuracies.

## Code
We proposed a broad family of models that incorporate global and structural information in various ways. This repository provides an implementation of both each individual model (in `models`) and a library for combining these into arbitrary configurations (including the Sandwich models described in the paper, in `running`) for the purposes of joint localization and repair tasks. This framework is generally applicable to any task that transforms code tokens to states that are useful for downstream tasks.

Since the models in this paper rely heavily on the presence of "edge" information (e.g., relational data such as data-flow connections), we also provide a library for reading such data from our own JSON format and providing it to the various models. These files additionally contain meta-information for the task of interest in this paper (joint localization and repair), for which we provide an output layer and train/eval optimizer loop. These components are contingent on the data release [status](#status).

### Configuration
The following parameters ought to be held fixed for all models, most of which are set correctly by default in config.yml:

- Where possible, all models are run on a single *NVidia RTX Titan GPU* with 24GB of memory. If not the case, this should be noted.<sup>1</sup>

<sup>1</sup>: This affects both the timing (of lesser interest), and the batch size, which is strongly dictated by GPU memory and can make a large difference in ultimate performance (which is also why it is explicitly reported).

The following results and variables should be reported for each run:

- The highest *joint localization & repair* accuracy reached in 100 steps (the key metric) on the full 'eval' dataset, using the model that performed highest in this metric on the heldout data at training time. For completeness, please also report the corresponding no-bug prediction accuracy (indicates false alarm rate), bug localization accuracy, and bug repair accuracy (between parentheses).
- *Details pertaining to the run*: the specific step at which that accuracy was achieved, the time taken per step, and the total number of parameters used by the model (printed at the start of training).
- The *hyper-parameters for this model*: at least, the maximum batch size in terms of total tokens (batchers are grouped by similar sample size for efficiency -- users are encouraged to use the default (12,500) for comparability), the learning rate, and any details pertaining to the model architecture (new innovations, with paper, are encouraged!) and its dimensions.
