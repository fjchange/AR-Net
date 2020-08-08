# AR-Net
Reproduction of ICME 2020 Weakly Supervised Video Anomaly Detectionn via Center-Guided Discriminative Learning.

With I3D RGB+Flow features, I get better `AUC 93.5% but with 1.3% FAR` with 600 epochs. Better AUC but poor FAR. **Not match to the Paper Reported**
|Source|Feature|AUC(%)|FAR(%)|
|----|---|---|---|
|Reproduced|I3D RGB+Flow|93.7%|1.3%|
|Paper|I3D RGB+Flow |91.4% |0.1%| 
As training going on, the AUC going up and FAR also going up. Waitting for the author's response.


## 1. Feature Preparation
1. Resize frame/flow to 224x224 and extract feature with I3D model pretrained on Kinetics-400 with 16 frames per clip.
2. To assure the length of RGB feature and Flow feature is with same length, I add an static flow frame at the end of Flows.

## 2. Traing and Testing
`python SH_train.py`

