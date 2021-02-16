# Word-level language modeling RNN

### Warning
`batchify()` has been modified to use rows as batches, thus, batchify will output 20 rows of ~3000 columns if you're using 20 batch_size

Similar changes to `get_batch()`

---

To run training, just run

```bash
python3 main.py --cuda --lr=0.01 --batch_size=1024 --eval_batch_size=10 --nhid=100 --emsize=100 --seq_size=6 --epochs=20
```
##### (tweak the parameters how you please)

If you want to just find the training loss, etc, you can use the `--skip-train` arg to take it straight to the pre-trainined model.

```bash
python3 main.py --cuda --skip-train
```

To tie embedding and decoder weights, use `--tie` argument.

To generate text, you can run-
#### (Output to generated.txt)

```bash
python3 generate.py --cuda --seed=19
```


#### (please specify a new seed for a new output)

To check the spearman correlation, run

#### (Output to correlation.txt)

```bash
python3 measure_correlation.py
```
