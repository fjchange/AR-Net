# AR-Net
Reproduction of ICME 2020 Weakly Supervised Video Anomaly Detectionn via Center-Guided Discriminative Learning.

The FAR in the the paper should be `False Alarm Rate in Normal Videos`.

## 1. Feature Preparation
1. Resize frame/flow to 224x224 and extract feature with I3D model pretrained on Kinetics-400 with 16 frames per clip.
2. To assure the length of RGB feature and Flow feature is with same length, I add an static flow frame at the end of Flows.

## 2. Traing and Testing
`python SH_train.py`

