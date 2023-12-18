# Remaining Useful Life prediction with a Deep Self-Supervised Learning Approach

The aim of this research is to develop an innovative ML methodology (i.e. Self-Supervised Learning) that allows AI to learn from available data without external annotations, thus improving the results of prognostics in predictive maintenance despite the limitation of labelled data.

# Description

This work was carried out within the PREDICT project, involving Anass Akrim, Christian Gogu (Principal Investigator), Michel Salaun and Rob Vingerhoeds at the University of Toulouse (Institut Clément Ader and ISAE-SUPAERO DISC).

The idea behind this project is to to investigate whether pre-training DL models in a self-supervised way on unlabeled sensors data can be useful for downstream tasks in PHM (i.e. Remaining Useful Life estimation) with only Few-Shots Learning.

In this research, the issue of data scarcity in a fatigue damage prognostics problem is addressed. The interest is in estimating the RUL of aluminum panels (typical of aerospace structures) subject to fatigue cracks from strain gauge data.

A synthetic dataset is used, composed of a large unlabeled dataset (i.e. strain gauges data of structures before failure) for pre-training, and a smaller labeled dataset (i.e. strain gauges data of structures until failure) for fine-tuning. If you find this repository helpful, please cite our work:

```
@article{akrim2023self,
  title={Self-Supervised Learning for data scarcity in a fatigue damage prognostic problem},
  author={Akrim, Anass and Gogu, Christian and Vingerhoeds, Rob and Sala{\"u}n, Michel},
  journal={Engineering Applications of Artificial Intelligence},
  volume={120},
  pages={105837},
  year={2023},
  publisher={Elsevier}}
``` 

# Acknowledgements

◦ This work was partially funded by Occitanie region under the Predict project. This funding is gratefully acknowledged. 

◦ This work has been carried out on the supercomputers PANDO (ISAE Supaero, Toulouse) and Olympe (CALMIP, Toulouse, project n°21042). Authors are grateful to ISAE Supaero and CALMIP for the hours allocated to this project.
