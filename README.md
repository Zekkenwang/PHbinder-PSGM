---

# PHbinder & PSGM: A Peptide-HLA Interaction Framework

This is a cascaded framework for peptide-HLA binding prediction (PHBinder) and HLA pseudo-sequence generation (PSGM). The PHbinder model is designed to predict the binding affinity between peptides and their corresponding MHC molecules (HLA). The PSGM model generates candidate HLA-I pseudo-sequences based on a given peptide and maps them to a corresponding list of HLA-I alleles.

### Model Replication

First, clone this repository to your local machine:

```bash
git clone https://github.com/Zekkenwang/PHbinder-PSGM.git
cd PHbinder-PSGM
```

### Setting Up the Python Environment

It is highly recommended to use Conda or `venv` to create an isolated Python virtual environment to avoid package conflicts.

1. **Create a virtual environment using Conda:**

   ```bash
   conda create -n ph_binding_env python=3.9  # Python 3.9 or higher is recommended
   conda activate ph_binding_env
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Model Training and Usage

### PHbinder: Training and Usage

This section provides detailed instructions on how to replicate and use the PHbinder model. PHbinder includes a pre-training phase using ESM2+LoRA, which significantly improves training efficiency and model performance.

#### Training the PHbinder Model

To start the training process, run the following command:

```bash
python scripts/phbinder.py
```

**Expected Output:**

* The pre-trained model weights will be saved to `PHbinder-PSGM/models/phbinder_pretrain`.
* The final PHbinder model weights will be saved to `PHbinder-PSGM/models/phbinder`.
* After the script finishes, the final evaluation metrics (Accuracy, F1 Score, Recall, Precision, MCC, AUC) on the test set will be printed to the console.

#### Predicting Epitopes with the PHbinder Model

To predict whether a peptide is an epitope using the trained model, run:

```bash
python scripts/predict_phbinder.py
```

This will evaluate the model on an external dataset. We have provided an example evaluation using the dataset in `external_data`. If you wish to use your own dataset, please update the `EXTERNAL_DATA_PATH` and `PREDICTION_OUTPUT_PATH` in the `phbinder_config` to your desired input and output paths.

### PSGM: Training and Usage

#### Training the PSGM Model

To train the PSGM model, execute the following command:

```bash
python scripts/train_psgm.py
```

**Expected Output:**

* The PSGM model weights will be saved to `PHbinder-PSGM/models/psgm`.

#### Generating HLA-I Pseudo-Sequences with the PSGM Model

Use the trained PSGM model to generate HLA-I pseudo-sequences from a given list of peptides and map them to a corresponding list of HLA Alleles.

Run the following script for generation:

```bash
python scripts/predict_psgm.py
```

This will generate predictions on an external dataset, with an example provided in `external_data`. To evaluate on your own data, modify the `EXTERNAL_DATA_PATH` and `PREDICTION_OUTPUT_PATH` in the `psgm_config` file to your specific input and output file paths.
