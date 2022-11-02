# Remaining Useful Life prediction with a Deep Self-Supervised Learning Approach

This work was carried out within the PREDICT project, involving Anass Akrim, Christian Gogu (Principal Investigator), Michel Salaun and Rob Vingerhoeds at the University of Toulouse (Institut Clément Ader and ISAE-SUPAERO DISC).

The idea behind this project is to to investigate whether pre-training DL models in a self-supervised way on unlabeled sensors data can be useful for downstream tasks in PHM (i.e. Remaining Useful Life estimation) with only Few-Shots Learning.

In this research, the issue of data scarcity in a fatigue damage prognostics problem is addressed. The interest is in estimating the RUL of aluminum panels (typical of aerospace structures) subject to fatigue cracks from strain gauge data.

A synthetic dataset is used, composed of a large unlabeled dataset (i.e. strain gauges data of structures before failure) for pre-training, and a smaller labeled dataset (i.e. strain gauges data of structures until failure) for fine-tuning.

# Acknowledgements

◦ This work was partially funded by Occitanie region under the Predict project. This funding is gratefully acknowledged. 

◦ This work has been carried out on the supercomputers PANDO (ISAE Supaero, Toulouse) and Olympe (CALMIP, Toulouse, project n°21042). Authors are grateful to ISAE Supaero and CALMIP for the hours allocated to this project.
