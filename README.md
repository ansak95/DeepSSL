# Remaining Useful Life prediction with a Deep Self-Supervised Learning Approach

The idea behind this project is to to investigate whether pre-training DL models in a self-supervised way on unlabeled sensors data can be useful for downstream tasks in PHM (i.e. Remaining Useful Life estimation) with only Few-Shots Learning.

In this research, the issue of data scarcity in a fatigue damage prognostics problem is addressed. The interest is in estimating the RUL of aluminum panels (typical of aerospace structures) subject to fatigue cracks from strain gauge data.

A synthetic dataset is used, composed of a large unlabeled dataset (\textit{i.e.} strain gauges data of structures before failure) for pre-training, and a smaller labeled dataset (\textit{i.e.} strain gauges data of structures until failure) for fine-tuning.


