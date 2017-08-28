import os
import json

import pandas as pd


try:
    CONFIG = json.load(open('config.json', 'r'))
except FileNotFoundError:
    CONFIG = json.load(open('default_config.json', 'r'))

DATA_ROOT = CONFIG['data_root']
CN_CONFIG_PATH = os.path.join(DATA_ROOT, CONFIG['forms']['b']['column_annotation_subpath'])
FLAT_FORMS_PATH = os.path.join(DATA_ROOT, CONFIG['forms']['b']['flat_data_subpath'])
ENCODED_EXTRACTION_PATH = os.path.join(DATA_ROOT, CONFIG['forms']['b']['encoded_extraction_subpath'])
EXTRACTION_PATH = os.path.join(DATA_ROOT, CONFIG['forms']['b']['extraction_subpath'])
FORMS_JSON_FOLDER = os.path.join(DATA_ROOT, CONFIG['forms']['json_data_subfolder'])
ACFU_MODEL_PATH = os.path.join(DATA_ROOT, CONFIG['forms']['b']['acfu_model_subpath'])
GRAPHS_FOLDER = os.path.join(DATA_ROOT, CONFIG['graphs_subfolder'])

CASE_ID_CN = 'form-*-case-*-@case_id'
DATE_CN = 'form-*-case-*-@date_modified'
FORM_ID_CN = "_id"
COUNTRY_CN = 'form-*-case-*-update-*-country'
ACFU_ROUND_CN = 'form-*-case-*-update-*-acfu_round'
FORM_TYPE_CN = "form-*-@name"

BUCKETS = json.load(open(os.path.join(DATA_ROOT, CONFIG['forms']['b']['buckets_subpath']), 'r'))
ACFU_MODEL_FNS = json.load(open(os.path.join(DATA_ROOT, CONFIG['forms']['b']['acfu_model_features']), 'r'))

CN_CONFIG = json.load(open(CN_CONFIG_PATH, 'r'))

LIFETIME_INPUT_CNS = CN_CONFIG['curated']['source']['lifetime'].copy()
assert not pd.Series(LIFETIME_INPUT_CNS).duplicated().any()
CURRENT_FORM_INPUT_CNS = CN_CONFIG['curated']['source']['current_form'].copy()
assert not pd.Series(CURRENT_FORM_INPUT_CNS).duplicated().any()
META_CNS = CN_CONFIG['curated']['source']['meta'].copy()
assert not pd.Series(META_CNS).duplicated().any()
NGRAM_CNS = CN_CONFIG['curated']['source']['ngram'].copy()
assert not pd.Series(NGRAM_CNS).duplicated().any()
INFANT_HIV_CNS = CN_CONFIG['curated']['source']['infant_hiv'].copy()
assert not pd.Series(INFANT_HIV_CNS).duplicated().any()

CURATED_SOURCE_CNS = LIFETIME_INPUT_CNS.copy()
CURATED_SOURCE_CNS += CURRENT_FORM_INPUT_CNS
CURATED_SOURCE_CNS += META_CNS
CURATED_SOURCE_CNS += NGRAM_CNS
CURATED_SOURCE_CNS += INFANT_HIV_CNS
CURATED_SOURCE_CNS = list(set(CURATED_SOURCE_CNS))

DOB_INPUT_CNS = CN_CONFIG['curated']['type']['dob'].copy()
assert all([cn in CURATED_SOURCE_CNS for cn in DOB_INPUT_CNS])
META_DATE_CNS = CN_CONFIG['curated']['type']['meta_date'].copy()
assert all([cn in CURATED_SOURCE_CNS for cn in META_DATE_CNS])
EVENT_DATE_INPUT_CNS = CN_CONFIG['curated']['type']['event_date'].copy()
assert all([cn in CURATED_SOURCE_CNS for cn in EVENT_DATE_INPUT_CNS])
DENSE_CONTINUOUS_CNS = CN_CONFIG['curated']['type']['dense_continuous'].copy()
assert all([cn in CURATED_SOURCE_CNS for cn in DENSE_CONTINUOUS_CNS])
OTHER_BUCKET_CNS = CN_CONFIG['curated']['type']['bucketise'].copy()
assert all([cn in CURATED_SOURCE_CNS for cn in OTHER_BUCKET_CNS])
