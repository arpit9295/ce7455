### Note
- Outputs are not shown on the shell, instead use `tail -50f ./logs/out.txt` or similar to observe outputs
- `main.py` uses an args parser, so feel free to use the arguments specified below
- Code assumes current working directory is `assignment-2`
- The eval_every behaviour has been changed from x iterations to x epochs
- Code uses an early stopping function

### Initialization

To fetch necessary data, run `python3 init.py`


### Run

```bash
nohup python3 main.py --eval-every=2 & tail -50f ./logs/out.txt
```

### Arguments available

```
--train TRAIN         Path to train file
--dev DEV             Path to test file
--test TEST           Path to dev file
--tag_scheme TAG_SCHEME
                      BIO or BIOES
--char-dim CHAR_DIM   Char embedding dimension
--word-dim WORD_DIM   Token embedding dimension
--word-lstm-dim WORD_LSTM_DIM
                      Token LSTM hidden layer size
--word-bidirect WORD_BIDIRECT
                      Use a bidirectional LSTM for words
--embedding-path EMBEDDING_PATH
                      Location of pretrained embeddings
--all-emb ALL_EMB     Load all embeddings
--crf CRF             Use CRF (0 to disable)
--dropout DROPOUT     Droupout on the input (0 = no dropout)
--epochs EPOCHS       Number of epochs to run
--weights WEIGHTS     path to Pretrained for from a previous run
--gradient-clip GRADIENT_CLIP
--output-dir OUTPUT_DIR
                      Stores data, plots and model files in this folder
--plot-every PLOT_EVERY
                      Plot after every PLOT_EVERY iterations
--eval-every EVAL_EVERY
                      Evaluate at the end of EVAL_EVERY epochs
```
