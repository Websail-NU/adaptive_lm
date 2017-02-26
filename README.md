This repository is an implementation an RNNLM using Tensorflow (r1.0).


## Usage

### Preparing data
We have a script to download and preprocess public LM dataset. Please see shell script files in [data](data/). For other corpus, you need to prepare `train.txt`, `valid.txt`, and `test.txt` and run the main preprocessing file in [preprocessing module](adaptive_lm/preprocess/).

### Training

You can train a langauge model with default option with:

```
python run_lm.py --training --save_config_file train_config.json
```

It will create a directory `experiments` and save all checkpoints and logs in the directory. By default, the script will use LSTM cell and train on PTB dataset. For other option, please add `--help` option.

### Testing

The same file can also be used for testing. To reuse the configuration file by passing `--load_config_filepaht` and override the configuration by provding new ones. For example

```
python run_lm.py --load_config_filepaht experiments/train_config.json --no-training
```

## Extending
There are many levels of modification in the code.
- `feed_dict` and `fetch`: implement new functions to modify default `feed_dict` and `fetch` dictionary from [default method](adaptive_lm/models/basic_rnnlm.py#L66), see [collecting token loss](run_lm.py#L13) for example. Note that `feed_dict` is mapped from grap node dictionary and data iterator's batch of the same keys (see [map_feeddict(batch, model_feed)](adaptive_lm/utils/run.py#L109)).
- Initialization and/or minor architecture changes: implement a new [BasicRNNHelper](adaptive_lm/models/rnnlm_helper.py#L3).
- New architecture: implement a new [RNNLM class](adaptive_lm/models/rnnlm.py#L7) (See [BasicRNNLM](adaptive_lm/models/basic_rnnlm.py) for example).

## TODO
- Support other cell types: add commandline argurment and improve [feed_state(.)](adaptive_lm/utils/run.py#L103)
- Provide decoder interface
- Change to Tensorflow r1.1 (contrib.rnn is no longer supported)
