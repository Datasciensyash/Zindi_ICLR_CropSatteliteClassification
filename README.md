## Datasciensyash code for ICLR Workshop Challenge #2: Radiant Earth Computer Vision for Crop Detection from Satellite Imagery.

**Original competition data must be stored in data\ folder.**
The original data is too large for me to send along with the solution.
Data folder structure has not been changed:
```
data\
	00\
		...
	01\
		...
	...
```

Packages, used in this solution:
```
tifffile
numpy
pandas
pytorch
matploitlib
lightgbm
```

Files in this solution:
```
Pipeline-DataPreparation.ipynb - Data preparation
Pipeline-ModelTrain.ipynb - Model training and creation of the submission file.
modules\
	dataset.py - Pytorch dataset class
	loss.py - Pytorch implementation of focal loss
	functional.py - Pytorch implementation of focal loss (as function, used un loss.py)
models\
	scn.py - Pytorch model of siam convolutional classifier.
```

Reproducibility:
```
Step 1: Pipeline-DataPreparation.ipynb -> Run all cells.
Step 2: Pipeline-ModelTrain.ipynb-> Run all cells.
Warning: For model training in step 2, i have used GPU instances. If you prefer to preform train on CPU, you need to change 'DEVICE' at settings dict to torch.device('cpu').

Submission filename: final-submission.csv
```
