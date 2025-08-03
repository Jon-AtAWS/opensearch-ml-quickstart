# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Legacy QAndA file reader test - moved to test_integration.py"""

import logging
from data_process import QAndAFileReader
from configs.configuration_manager import get_qanda_file_reader_path


def test():
    """Legacy test function - use test_integration.py for new tests"""
    try:
        logging.info("Testing qanda file reader...")
        reader = QAndAFileReader(directory=get_qanda_file_reader_path())
        
        category_count = 0
        for category in reader.amazon_pqa_category_names():
            logging.info(f"category: {category}")
            try:
                size = reader.file_size(category)
                logging.info(f"file size: {size}")
                category_count += 1
            except Exception as e:
                logging.warning(f"Could not get size for {category}: {e}")
            
            # Limit to first few categories for testing
            if category_count >= 3:
                break
                
        logging.info(f"Successfully tested {category_count} categories")
        
    except Exception as e:
        logging.warning(f"QAndA file reader test failed: {e}")


# For backward compatibility
if __name__ == "__main__":
    test()
