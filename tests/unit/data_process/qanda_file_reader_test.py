# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from data_process import QAndAFileReader
from configs import get_config, QANDA_FILE_READER_PATH


def test():
    logging.info("Testing qanda file reader...")
    reader = QAndAFileReader(directory=QANDA_FILE_READER_PATH)
    for category in reader.amazon_pqa_category_names():
        logging.info(f"category: {category}")
        logging.info(f"file size: {reader.file_size(category)}")
