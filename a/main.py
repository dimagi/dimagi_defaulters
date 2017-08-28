from typing import List

import pandas as pd

import constants
from . import extract
from . import describe


def historical_extract_flat_forms(persist=False) -> None:
    vdescs = describe.describe_objects(jsons_path=constants.FORMS_JSON_FOLDER)
    vdescs = extract.discard_vars(vdescs)
    df = extract.get_flat(jsons_path=constants.FORMS_JSON_FOLDER, vdescs=vdescs)
    if persist:
        print('--> Saving result to path: {}'.format(constants.FLAT_FORMS_PATH))
        df.to_csv(constants.FLAT_FORMS_PATH, index=False)
    return df


def live_extract_flat_forms(form_dicts: List[dict]) -> pd.DataFrame:
    vdescs = describe.describe_objects(form_dicts=form_dicts)
    vdescs = extract.discard_vars(vdescs)
    df = extract.get_flat(form_dicts=form_dicts, vdescs=vdescs)
    return df
