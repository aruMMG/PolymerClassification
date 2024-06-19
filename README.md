# Adapting the Transformer Model for Recycling Plastic Sorting
Transformer based model for recycling plastic classification using Fourier Transform Infrared (FTIR) data. 
![Alt text](https://github.com/aruMMG/PolymerClassification/blob/main/asset/overview.jpg?raw=true "Title")
## Results
![Alt text](https://github.com/aruMMG/PolymerClassification/blob/main/asset/accuracy.jpg?raw=true "Title")
Compares the accuracy of two neural network architectures—CNN and our proposed Transformer-based model—under different conditions on test data. The "Base" is the initial model without any additional preprocessing module. Then the performance evaluated with additioanl preprocessing modules: average pooling (Base + Avg), layer normalisation (Base + LN), and combination of all preprocessing steps (Base + Pre). The dashed line shows the models' performance without baseline correction. The proposed preprocessing module significantly improve the models' performance and especially the proposed Transformer-based model outperforms the CNN in all different conditions.
## Getting started
To getting started plese follow:

```console
git clone https://github.com/aruMMG/PolymerClassification.git
cd PolymerClassification
pip isntall -r requirements.txt
```

For training the Transformer model use the below steps. This will train for ten folds, save a argument.txt file and save a folder named "data_split" stores txt files with training and testing data split. This data split txt files can be used for consistent model training.:

```console
python train_csv.py --data_dir path/to/data/dir --baseline
```
For using a data split use the following commant to train other models including CNN, Fully connected models.
```console
python train_from_txt.py --data_dir path/to/data/dir --baseline --log_name path/to/directory/containing/data_split --model FC
```