# FER-CAPSULENET-FER2013
**Usage**
**Step 1. Install libraries**
```
conda env create -f environment.yml
```
**Step 2. Download the original fer2013 csv file from this link**
```
https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data
```
**Step 3. Put the fer2013 csv file in the main folder**

**Step 4. Download the pre-trained models from this link**
```
svm_with_HE.ipynb
```
**Step 5. Put the pre-trained models in the following folder**
```
pre_trained_models
```
**To train the model from scratch, run the following**
```
For baseline model: python capsulenet_baseline.py
For batch_norm model: python capsulenet_baseline+batch_norm.py
For 2_conv_layers model: python capsulenet_2_conv_layers.py
For 3_conv_layers model: python capsulenet_3_conv_layers.py
```

**After training the new model will be saved at the result folder**

**For evaluation on the pre-trained models, run the following**
```
For baseline model: python capsulenet_baseline.py -t -w pre_trained_models/trained_baseline_model.h5
For batch_norm model: python capsulenet_baseline+batch_norm.py -t -w pre_trained_models/trained_baseline+batch_norm_model.h5
For 2_conv_layers model: python capsulenet_2_conv_layers.py -t -w pre_trained_models/trained_2conv_model.h
For 3_conv_layers model: python capsulenet_3_conv_layers.py -t -w pre_trained_models/trained_3conv_model.h5
```

**For evaluation on your model, run the following**
```
 python "your_model".py -t -w result/"your_model h5 file"

```
