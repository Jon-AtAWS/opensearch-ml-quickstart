# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import json


def get_base_mapping(base_mapping_path):
    with open(base_mapping_path, "r") as file:
        data = json.load(file)
    return data


def mapping_update(base_mapping, settings):
    for key, value in settings.items():
        if (
            key in base_mapping
            and isinstance(base_mapping[key], dict)
            and isinstance(value, dict)
        ):
            mapping_update(base_mapping[key], value)
        else:
            base_mapping[key] = value
