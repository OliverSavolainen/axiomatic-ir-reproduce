# Reproducibility study of "Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models" by Oliver Savolainen and Dur E Najaf Amjad

This is the repository for our reproducibility study on the paper "Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models". The repository has been created based on their original repository: https://github.com/catherineschen/axiomatic-ir-interventions

The following 3 sections are from the README from that repository:


By Catherine Chen, Jack Merullo, and Carsten Eickhoff (Brown University, University of Tübingen)

This code corresponds to the paper: __Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models__, in _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24)_, July 14–18, 2024, Washington, DC, USA. [Link to paper](https://arxiv.org/abs/2405.02503)

# Setup

This code uses a copy of the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) package, which we make additional changes to support activation patching in a retrieval setting. Changes made to the original TransformerLens package to support activation patching for retrieval (TAS-B) can be found in the following files:

- `components.py`
- `loading_from_pretrained.py`
- `HookedEncoder.py`


# Demo

We provide a demo of how we perform activation patching for retrieval in `retrieval_patching_demo.py`. 

```
python retrieval_patching_demo.py
```

## Experiments

To run the patching experiments:

```
python experiment.py EXPERIMENT_TYPE PERTURB_TYPE
```

The patching experiments are currently designed to be run on a single GPU, and depending on the experiment, can take several hours to days to complete. To save time, we would suggest setting up separate jobs for each type of patching experiment.

`EXPERIMENT_TYPE`:
- `block`: patch blocks for each layer over individual token positions (residual stream, attention block, MLP)
- `head_all`: patch individual attention heads over all token positions
- `head_pos`: patch target attention heads over individual token positions
- `head_attn`: get attention head patterns for target heads
- `labels`: get tokenized documents

`PERTURB_TYPE`:
- `append`: target query term added to the end of a document
- `prepend`: target query term added to the beginning of a document


To visualize the results:
```
python plot_results.py EXPERIMENT_TYPE PERTURB_TYPE
```

-----------
Corresponding author: Catherine Chen (catherine_s_chen [at] brown [dot] edu)

If you find this code or any of the ideas in the paper useful, please consider citing:
```
@inproceedings{chen2024axiomatic,
  title={Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models},
  author={Chen, Catherine and Merullo, Jack and Eickhoff, Carsten},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1401--1410},
  year={2024}
}
```

# Extensions

We have worked on 2 extensions:

1) mMARCO Spanish dataset

To create the dataset:

```
python load_dataset.py
```

The dataset created by us can be found from /data_mmarco_spanish.

To run experiments with this dataset:

```
python experiment.py EXPERIMENT_TYPE PERTURB_TYPE
``` 

To visualize the results:
```
python plot_results_mmarco.py EXPERIMENT_TYPE PERTURB_TYPE
```

2) LNC1 axiom

To choose the 33 top impact queries from the original dataset:

```
python top_queries_lnc1.py PERTURB_TYPE
```

The files created from this can be found in /data.

To run experiments with this dataset:

```
python experiment_lnc1.py EXPERIMENT_TYPE PERTURB_TYPE
``` 

To visualize the results:
```
python plot_results_lnc1.py EXPERIMENT_TYPE PERTURB_TYPE
```

To see the score differences between different perturbations:

```
python perturbation_comp.py PERTURB_TYPE
```