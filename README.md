# Generating Multi-type Temporal Sequences to Mitigate Class-imbalanced Problem

This repo contains the code and data of paper [Generating Multi-type Temporal Sequences to Mitigate Class-imbalanced Problem](https://arxiv.org/abs/2104.03428), submitted to [ECML-PKDD 2021](https://2021.ecmlpkdd.org/).

`data/` folder contains the synthetic dataset. The positive and negative sequences are corresponding to the positive and negative dataset in the paper.

`models/` folder contains the saved Keras model weights.

`notebooks/` folder contains the Jupyter notebooks for code demo.

`sgtlstm/` folder contains Python code source files of this project. Especially, the `SeqGan.py` and `TimeLSTM.py` have our implementations of [SeqGan](https://arxiv.org/abs/1609.05473) and [TimeLSTM](https://www.ijcai.org/proceedings/2017/504).

For collaboration and questions, please email the arthuros at _{lun,nimas,zhuo,andrew.cohen}@unity3d.com_