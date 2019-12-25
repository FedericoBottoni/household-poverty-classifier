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
python auto_ml.py
```

## Train & Test

You can train the network and evaluate the metrics :

```bash
python train.py
```

## Authors

**Mattia Artifoni**

- [github/m-artifoni](https://github.com/m-artifoni)

**Federico Bottoni**

- [github/FedericoBottoni](https://github.com/federicobottoni)

**Riccardo Capra**

- [github/riccardocapra](https://github.com/riccardocapra)

## License
