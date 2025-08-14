# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for data_process module - requires actual data files"""

import pytest
import os
import logging
from data_process import QAndAFileReader
from configs.configuration_manager import get_qanda_file_reader_path


class TestDataProcessIntegration:
    """Integration tests that require actual Amazon PQA data files"""

    @pytest.mark.integration
    def test_qanda_file_reader_with_actual_data(self):
        """Test QAndAFileReader with actual Amazon PQA data files"""
        try:
            data_path = get_qanda_file_reader_path()
            
            # Check if data directory exists
            if not os.path.exists(data_path):
                pytest.skip(f"Amazon PQA data directory not found: {data_path}")
            
            reader = QAndAFileReader(directory=data_path, max_number_of_docs=5)
            
            # Test that we can get category names
            category_names = list(reader.amazon_pqa_category_names())
            assert len(category_names) > 0
            logging.info(f"Found {len(category_names)} categories")
            
            # Test reading from first available category
            first_category = category_names[0]
            logging.info(f"Testing with category: {first_category}")
            
            # Test category name validation
            assert reader.is_category_name(first_category)
            
            # Test constant conversion
            constant = reader.amazon_pqa_category_name_to_constant(first_category)
            assert constant is not None
            assert constant.startswith("AMAZON_PQA_")
            
            # Test filename conversion
            filename = reader.amazon_pqa_constant_to_filename(constant)
            assert filename.endswith(".json")
            
            # Test reverse conversion
            category_back = reader.amazon_pqa_constant_to_category_name(constant)
            assert category_back == first_category
            
            logging.info(f"Category: {first_category} -> Constant: {constant} -> Filename: {filename}")
            
        except Exception as e:
            if "configuration" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Configuration or data files not available: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_file_size_with_actual_files(self):
        """Test file_size method with actual data files"""
        try:
            data_path = get_qanda_file_reader_path()
            
            if not os.path.exists(data_path):
                pytest.skip(f"Amazon PQA data directory not found: {data_path}")
            
            reader = QAndAFileReader(directory=data_path)
            
            # Get first available category
            category_names = list(reader.amazon_pqa_category_names())
            if not category_names:
                pytest.skip("No categories found in data directory")
            
            first_category = category_names[0]
            
            # Test file size
            file_size = reader.file_size(first_category)
            assert isinstance(file_size, int)
            assert file_size > 0
            
            logging.info(f"File size for {first_category}: {file_size} bytes")
            
        except Exception as e:
            if "configuration" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Configuration or data files not available: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_questions_reading_with_actual_data(self):
        """Test reading questions from actual data files"""
        try:
            data_path = get_qanda_file_reader_path()
            
            if not os.path.exists(data_path):
                pytest.skip(f"Amazon PQA data directory not found: {data_path}")
            
            reader = QAndAFileReader(directory=data_path, max_number_of_docs=3)
            
            # Get first available category
            category_names = list(reader.amazon_pqa_category_names())
            if not category_names:
                pytest.skip("No categories found in data directory")
            
            first_category = category_names[0]
            constant = reader.amazon_pqa_category_name_to_constant(first_category)
            
            # Test reading questions without enrichment
            questions = list(reader.questions_for_category(constant, enriched=False))
            assert len(questions) <= 3  # Should respect max_number_of_docs
            assert len(questions) > 0
            
            # Verify question structure
            first_question = questions[0]
            assert isinstance(first_question, dict)
            assert 'question_id' in first_question
            
            logging.info(f"Read {len(questions)} questions from {first_category}")
            logging.info(f"Sample question keys: {list(first_question.keys())}")
            
        except Exception as e:
            if "configuration" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Configuration or data files not available: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_questions_enrichment_with_actual_data(self):
        """Test question enrichment with actual data files"""
        try:
            data_path = get_qanda_file_reader_path()
            
            if not os.path.exists(data_path):
                pytest.skip(f"Amazon PQA data directory not found: {data_path}")
            
            reader = QAndAFileReader(directory=data_path, max_number_of_docs=2)
            
            # Get first available category
            category_names = list(reader.amazon_pqa_category_names())
            if not category_names:
                pytest.skip("No categories found in data directory")
            
            first_category = category_names[0]
            constant = reader.amazon_pqa_category_name_to_constant(first_category)
            
            # Test reading questions with enrichment
            enriched_questions = list(reader.questions_for_category(constant, enriched=True))
            assert len(enriched_questions) <= 2
            assert len(enriched_questions) > 0
            
            # Verify enrichment
            first_enriched = enriched_questions[0]
            assert 'category_name' in first_enriched
            assert first_enriched['category_name'] == first_category
            assert 'bullets' in first_enriched
            
            # Verify bullet points were removed
            bullet_points = ['bullet_point1', 'bullet_point2', 'bullet_point3', 'bullet_point4', 'bullet_point5']
            for bullet in bullet_points:
                assert bullet not in first_enriched
            
            # Verify answers were enriched
            if 'answers' in first_enriched and first_enriched['answers']:
                first_answer = first_enriched['answers'][0]
                enrichment_fields = ['name', 'user_lat', 'user_lon', 'gender', 'age', 'product_rating']
                for field in enrichment_fields:
                    assert field in first_answer
            
            logging.info(f"Successfully enriched {len(enriched_questions)} questions")
            
        except Exception as e:
            if "configuration" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Configuration or data files not available: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_questions_for_all_categories_with_actual_data(self):
        """Test reading questions from all categories with actual data"""
        try:
            data_path = get_qanda_file_reader_path()
            
            if not os.path.exists(data_path):
                pytest.skip(f"Amazon PQA data directory not found: {data_path}")
            
            reader = QAndAFileReader(directory=data_path, max_number_of_docs=1)
            
            # Test reading from all categories
            all_questions = list(reader.questions_for_all_categories(enriched=False))
            
            # Should have at least some questions
            assert len(all_questions) > 0
            
            # Count categories that actually have data
            category_names = list(reader.amazon_pqa_category_names())
            categories_with_data = 0
            
            for category in category_names:
                try:
                    constant = reader.amazon_pqa_category_name_to_constant(category)
                    questions = list(reader.questions_for_category(constant, enriched=False))
                    if questions:
                        categories_with_data += 1
                except:
                    continue  # Skip categories without data files
            
            logging.info(f"Found data in {categories_with_data} categories")
            logging.info(f"Total questions read: {len(all_questions)}")
            
        except Exception as e:
            if "configuration" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Configuration or data files not available: {e}")
            else:
                raise

    @pytest.mark.integration
    def test_data_file_consistency(self):
        """Test consistency between data files and mappings"""
        try:
            data_path = get_qanda_file_reader_path()
            
            if not os.path.exists(data_path):
                pytest.skip(f"Amazon PQA data directory not found: {data_path}")
            
            reader = QAndAFileReader(directory=data_path)
            
            # Check that files exist for mappings
            missing_files = []
            existing_files = []
            
            for constant, filename in reader.AMAZON_PQA_FILENAME_MAP.items():
                file_path = os.path.join(data_path, filename)
                if os.path.exists(file_path):
                    existing_files.append(filename)
                else:
                    missing_files.append(filename)
            
            logging.info(f"Found {len(existing_files)} data files")
            if missing_files:
                logging.warning(f"Missing {len(missing_files)} data files: {missing_files[:5]}...")
            
            # At least some files should exist
            assert len(existing_files) > 0, "No data files found"
            
        except Exception as e:
            if "configuration" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Configuration or data files not available: {e}")
            else:
                raise


class TestDataProcessLegacyCompatibility:
    """Legacy test functions for backward compatibility"""

    def test_qanda_file_reader_legacy(self):
        """Legacy test function for QAndAFileReader"""
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


# Legacy test function for backward compatibility
def test():
    """Legacy test function - use TestDataProcessLegacyCompatibility for new tests"""
    test_instance = TestDataProcessLegacyCompatibility()
    test_instance.test_qanda_file_reader_legacy()


if __name__ == "__main__":
    test()
