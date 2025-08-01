# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from data_process import QAndAFileReader
from configs.configuration_manager import get_raw_config_value, get_qanda_file_reader_path


def test():
    logging.info("Testing qanda file reader...")
    reader = QAndAFileReader(directory=get_qanda_file_reader_path())
    for category in reader.amazon_pqa_category_names():
        logging.info(f"category: {category}")
        logging.info(f"file size: {reader.file_size(category)}")
