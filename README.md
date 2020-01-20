# household-poverty-classifier

> The project is an implementation for solve the Costa Rican household poverty level prediction. This is an implementation for the [Kaggle project](https://www.kaggle.com/c/costa-rican-household-poverty-prediction)

## Requirements

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Data cleaning

You can re-execute the script to export the final dataset:

```bash
python preprocess_raw.py
```

## AutoML

You can sample the best HPs :

```bash
python sample_hps.py ./data/train.csv
```

## Train & Test

You can train the network and evaluate the metrics :

```bash
python  nnscore.py ./data/train.csv
```

You can train the network from data with some features encoded manually and evaluate the metrics :

```bash
python  nnscore.py ./data/train_enc.csv
```

## Fairness check

You can get the predictions' frequencies of the classes grouped by some defined feature to check the fairness of the model:

```bash
python fairness.py ./data/train.csv.py
```

## Plot

You can plot the NN history by executing the Ipython notebook /plot/plot.ipynb along with /data/train.cvs on Google Colab.

## Authors

**Mattia Artifoni**

- [github/m-artifoni](https://github.com/m-artifoni)

**Federico Bottoni**

- [github/FedericoBottoni](https://github.com/federicobottoni)

**Riccardo Capra**

- [github/riccardocapra](https://github.com/riccardocapra)

## License
