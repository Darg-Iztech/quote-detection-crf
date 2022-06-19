# Quote Detection CRF
Conditional Random Field (CRF) model for sequence labeling as the baseline model for Quote Detection task.

### Run CRF baseline:

```bash
python qtask_baseline.py
```

### Calculate ROUGE scores:

```bash
pip install rouge

rouge -f data/pred_quotes.txt data/true_quotes.txt --avg
```

### Results on PQN mode:

| Metric | Score   |
| ------ | ------- |
| F1     | 0.27300 |
| R1     | 0.07595 |
| R2     | 0.04905 |
| RL     | 0.06980 |

### References:

https://github.com/arielsho/SemEval-2020-Task-5

https://github.com/pltrdy/rouge
