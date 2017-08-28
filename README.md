# DataProphet - Dimagi - Mothers2Mothers

## Get started

### Obtain source code

`git clone git@github.com:dataprophet/dimagi_m2m.git`

### Obtain data folder

DataProphet will deliver the data folder as a compressed archive upon completion of the project. Place a copy of the data folder on your file system and copy `default_config.py` to `config.py` and adjust the paths in that file to reflect the location of your data folder copy.

### Install dependencies

These Python modules are written for `Python 3.6`. You'll need the `pip` python package management tool to install dependencies. We recommend using `virtualenv` to install Python libraries in an environment that is isolated from the rest of your operating system.

Assuming you have `pip` installed and have activated your virtualenv (if you are using virtualenv), install the necessary dependencies using `pip`:

```bash
pip3 install wheel
pip3 install Cython
pip3 install ipython
pip3 install -r requirements.txt
```

Install xgboost as per [instructions](http://xgboost.readthedocs.io/en/latest/build.html) (don't miss the Python specific step, we recommend the route of simply updating your `PYTHONPATH`).

## Executing Python code

To run any of the below operations, first `cd` to the project root folder and start a python interpreter:

```bash
python3
```

or 

```bash
ipython3
```

Once inside the interpreter execute the statements as shown in the relevant section below.

## Predictive model

### Extract flat file

```python
import a
df = a.main.historical_extract_flat_forms(persist=True)
```

### Extract features

```python
import b
import constants
dfi = b.helpers.get_input_df(constants.CURATED_SOURCE_CNS, verbose=True)
dfi = b.helpers.clean(dfi, verbose=True)
meta = b.helpers.get_input_df(constants.META_CNS, verbose=True)
meta = b.helpers.clean(meta, verbose=True)
assert dfi.shape[0] == meta.shape[0]
rdf, rdfd, _, _ = b.curated.main.extract(dfi, meta, verbose=True)
# save to csv if you wish to re-use this extraction in latter sessions
```

### Train ACFU increment model

```python
import pickle
import b
import constants
rdf = pd.read_csv(constants.ENCODED_EXTRACTION_PATH, dtype=str)
mask = b.predict.acfu_increment_model_mask(rdfd, form_count=2, form_type='AN Appointment Diary', hiv_pos_only=False, sa_only=False)
X = rdfd.iloc[mask]
y_cn = 'will_acfu_incr_within_90_days'
for cn in constants.ACFU_MODEL_FNS + [y_cn]:
    X[cn] = X[cn].astype(float)
cns = constants.ACFU_MODEL_FNS + [y_cn, constants.CASE_ID_CN]
result = b.predict.main(X[cns], y_cn, plots=True, verbose=True)
# save to pickle file if you wish to re-use this model in latter sessions
# e.g. pickle.dump(result['gbm'], open('example_data/acfu_model_2nd_an_form.pkl, 'wb'))
```

### Predict for new json example

```python
import json
import main
import os
example_json_paths = next(os.walk('example_data/original_forms_json/'))[2]
forms = [json.load(open(os.path.join('example_data/original_forms_json', ejp), 'r')) for ejp in example_json_paths]
pred = main.get_live_prediction(forms)
```
