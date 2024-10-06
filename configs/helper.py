# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
from dotenv import load_dotenv
from .config import DEFAULT_ENV_PATH


def get_config(config_name, env_path=DEFAULT_ENV_PATH):
    load_dotenv(env_path)
    value = os.getenv(config_name)
    if value == "None":
        value = None
    return value


def validate_configs(configs, required_args):
    for required_arg in required_args:
        if required_arg not in configs or configs[required_arg] == None:
            raise ValueError(
                f"{required_arg} is missing or none, please specify {required_arg} in the configs"
            )


# if default value is none, the arg is required
def parse_arg_from_configs(configs, arg, default_value=None):
    if arg in configs:
        return configs[arg]
    elif default_value != None:
        return default_value
    else:
        raise ValueError(f"{arg} is missing, please specify {arg} in the configs")
