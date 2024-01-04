# 🚀 Deep Self-Supervised Learning for Remaining Useful Life Prediction

## Project Introduction
This repository showcases a cutting-edge approach to Remaining Useful Life (RUL) prediction, utilizing Deep Self-Supervised Learning. This innovative method represents a paradigm shift in Machine Learning, enabling AI systems to extract meaningful insights from available unlabelled data, without the need for externally provided annotations. This research addresses a critical challenge in predictive maintenance, particularly in environments where labeled data is scarce or difficult/expensive to obtain.

## 📖 Detailed Description

### Background
Initiated as part of the PREDICT project, this research represents a collaborative effort between experts at the University of Toulouse, including Institut Clément Ader and ISAE-SUPAERO DISC. The project focuses on leveraging the untapped potential of self-supervised learning in the domain of Prognostics and Health Management (PHM). Specifically, it aims to demonstrate the efficacy of pre-training Deep Learning models on large volumes of unlabeled sensor data and applying them to PHM tasks like RUL estimation, even with minimal labeled data availability.

### Research Focus
The core challenge tackled here is the scarcity of data in fatigue damage prognostics. The project's ambition is to accurately estimate the Remaining Useful Life of critical components, such as aluminum panels commonly used in aerospace structures, which are prone to fatigue cracks. To achieve this, the research utilizes strain gauge data, a type of data that presents unique challenges due to its nature and collection methods.


### Dataset Composition
A synthetic dataset forms the backbone of this research. It is strategically divided into two key components:
- A large, unlabeled dataset comprising strain gauge readings from structures prior to failure, used for the initial phase of model pre-training.
- A smaller, labeled dataset containing strain gauge data up to the point of structural failure, utilized for subsequent fine-tuning of the models.

### Contribution and Citation
The findings and methodologies developed in this project could be invaluable to researchers and practitioners in the field. Those who find this repository beneficial for their work are encouraged to cite the published research:

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

## 🙏 Acknowledgements

◦ This work was partially funded by Occitanie region under the Predict project. This funding is gratefully acknowledged. 

◦ This work has been carried out on the supercomputers PANDO (ISAE Supaero, Toulouse) and Olympe (CALMIP, Toulouse, project n°21042). Authors are grateful to ISAE Supaero and CALMIP for the hours allocated to this project.
