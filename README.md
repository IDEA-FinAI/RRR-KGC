This is the code implementation for RRR-KGC.

Before running the code, You should add you OpenAI Key in llm_api.py.

A single test can be done by running the command:

```python
python rrr_kgc.py --forward --fs_ctx --wiki_ctx --cand_ctx
```

We recommend to run the experiments by set the hyper-parameters in run_parallel.py, then run the command:

```python
python run_parallel.py
```
