# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import logging
from configs import get_config
from data_process import QAndAFileReader


def test():
    logging.info("Testing qanda file reader...")
    qanda_file_reader_path = get_config("QANDA_FILE_READER_PATH")
    reader = QAndAFileReader(directory=qanda_file_reader_path)
    for category in reader.amazon_pqa_category_names():
        logging.info(f"category: {category}")
        logging.info(f"file size: {reader.file_size(category)}")
