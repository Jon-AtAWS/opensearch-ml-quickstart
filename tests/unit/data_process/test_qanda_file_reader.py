# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for data_process.qanda_file_reader module"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, mock_open, MagicMock
from faker import Faker

from data_process.qanda_file_reader import QAndAFileReader


class TestQAndAFileReader:
    """Test cases for QAndAFileReader class"""

    def setup_method(self):
        """Setup method run before each test"""
        # Create a temporary directory for testing
        self.test_directory = "/test/directory"
        self.max_docs = 10
        
        # Sample test data
        self.sample_question = {
            "question_id": "test_id_1",
            "asin": "B001TEST",
            "question": "Test question?",
            "answers": [
                {
                    "answer": "Test answer 1",
                    "answer_time": "2023-01-01"
                },
                {
                    "answer": "Test answer 2", 
                    "answer_time": "2023-01-02"
                }
            ],
            "bullet_point1": "Feature 1",
            "bullet_point2": "Feature 2",
            "bullet_point3": "Feature 3",
            "bullet_point4": "Feature 4",
            "bullet_point5": "Feature 5",
            "product_description": "Test product",
            "brand_name": "Test Brand",
            "item_name": "Test Item"
        }

    def test_qanda_file_reader_init_default(self):
        """Test QAndAFileReader initialization with default values"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        assert reader.directory == self.test_directory
        assert reader.max_number_of_docs == -1
        assert isinstance(reader.asins, set)
        assert isinstance(reader.questions, set)
        assert len(reader.asins) == 0
        assert len(reader.questions) == 0
        assert reader.amazon_pqa_constants == reader.AMAZON_PQA_FILENAME_MAP.keys()
        assert isinstance(reader.fake, Faker)

    def test_qanda_file_reader_init_with_max_docs(self):
        """Test QAndAFileReader initialization with max_number_of_docs"""
        reader = QAndAFileReader(directory=self.test_directory, max_number_of_docs=self.max_docs)
        
        assert reader.directory == self.test_directory
        assert reader.max_number_of_docs == self.max_docs

    def test_amazon_pqa_category_name_to_constant(self):
        """Test amazon_pqa_category_name_to_constant method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock the AMAZON_PQA_CATEGORY_MAP
        with patch.object(reader, 'AMAZON_PQA_CATEGORY_MAP', {
            'jeans': 'AMAZON_PQA_JEANS',
            'monitors': 'AMAZON_PQA_MONITORS'
        }):
            assert reader.amazon_pqa_category_name_to_constant('jeans') == 'AMAZON_PQA_JEANS'
            assert reader.amazon_pqa_category_name_to_constant('monitors') == 'AMAZON_PQA_MONITORS'

    def test_amazon_pqa_constant_to_filename(self):
        """Test amazon_pqa_constant_to_filename method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Test with actual constants from the class
        assert reader.amazon_pqa_constant_to_filename('AMAZON_PQA_JEANS') == 'amazon_pqa_jeans.json'
        assert reader.amazon_pqa_constant_to_filename('AMAZON_PQA_MONITORS') == 'amazon_pqa_monitors.json'

    def test_amazon_pqa_constant_to_category_name(self):
        """Test amazon_pqa_constant_to_category_name method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock the AMAZON_PQA_CATEGORY_MAP
        with patch.object(reader, 'AMAZON_PQA_CATEGORY_MAP', {
            'jeans': 'AMAZON_PQA_JEANS',
            'monitors': 'AMAZON_PQA_MONITORS',
            'headphones': 'AMAZON_PQA_HEADPHONES'
        }):
            assert reader.amazon_pqa_constant_to_category_name('AMAZON_PQA_JEANS') == 'jeans'
            assert reader.amazon_pqa_constant_to_category_name('AMAZON_PQA_MONITORS') == 'monitors'
            assert reader.amazon_pqa_constant_to_category_name('NONEXISTENT') is None

    def test_amazon_pqa_category_names(self):
        """Test amazon_pqa_category_names method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock the AMAZON_PQA_CATEGORY_MAP
        test_categories = ['jeans', 'monitors', 'headphones']
        with patch.object(reader, 'AMAZON_PQA_CATEGORY_MAP', {
            'jeans': 'AMAZON_PQA_JEANS',
            'monitors': 'AMAZON_PQA_MONITORS',
            'headphones': 'AMAZON_PQA_HEADPHONES'
        }):
            category_names = list(reader.amazon_pqa_category_names())
            assert len(category_names) == 3
            for category in test_categories:
                assert category in category_names

    @patch('os.path.getsize')
    def test_file_size_success(self, mock_getsize):
        """Test file_size method with successful file size retrieval"""
        mock_getsize.return_value = 1024
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock the category mapping
        with patch.object(reader, 'AMAZON_PQA_CATEGORY_MAP', {'jeans': 'AMAZON_PQA_JEANS'}):
            size = reader.file_size('jeans')
            assert size == 1024
            mock_getsize.assert_called_once_with(f"{self.test_directory}/amazon_pqa_jeans.json")

    def test_file_size_unknown_category(self):
        """Test file_size method with unknown category"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock empty category mapping
        with patch.object(reader, 'AMAZON_PQA_CATEGORY_MAP', {}):
            with pytest.raises(KeyError, match="'unknown_category'"):
                reader.file_size('unknown_category')

    def test_filename_to_constant_name(self):
        """Test _filename_to_constant_name method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Test various filename transformations
        assert reader._filename_to_constant_name("./amazon_pqa_jeans.json") == "AMAZON_PQA_JEANS"
        assert reader._filename_to_constant_name("amazon_pqa_headlight_&_tail_light.json") == "AMAZON_PQA_HEADLIGHT_AND_TAIL_LIGHT"
        assert reader._filename_to_constant_name("amazon_pqa_in-dash_navigation.json") == "AMAZON_PQA_IN_DASH_NAVIGATION"

    def test_filename_to_constant_value(self):
        """Test _filename_to_constant_value method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Test various filename transformations
        assert reader._filename_to_constant_value("./amazon_pqa_jeans.json") == "jeans"
        assert reader._filename_to_constant_value("amazon_pqa_headlight_&_tail_light.json") == "headlight and tail light"
        assert reader._filename_to_constant_value("amazon_pqa_in-dash_navigation.json") == "in-dash navigation"

    @patch('glob.glob')
    def test_map_categories_to_constants(self, mock_glob):
        """Test _map_categories_to_constants method"""
        mock_glob.return_value = [
            "/test/directory/amazon_pqa_jeans.json",
            "/test/directory/amazon_pqa_monitors.json"
        ]
        
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            reader._map_categories_to_constants()
            
            # Verify print was called with expected format
            mock_print.assert_any_call("  AMAZON_PQA_CATEGORY_MAP = {")
            mock_print.assert_any_call("  }")

    @patch('glob.glob')
    def test_map_constants_to_filenames(self, mock_glob):
        """Test _map_constants_to_filenames method"""
        mock_glob.return_value = [
            "./amazon_pqa_jeans.json",
            "./amazon_pqa_monitors.json"
        ]
        
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            reader._map_constants_to_filenames()
            
            # Verify print was called with expected format
            mock_print.assert_any_call("  AMAZON_PQA_FILENAME_MAP = {")
            mock_print.assert_any_call("  }")

    @patch('random.random')
    def test_random_gender(self, mock_random):
        """Test random_gender method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Test "other" gender (> 0.95)
        mock_random.return_value = 0.96
        assert reader.random_gender() == "other"
        
        # Test "female" gender (> 0.475 and <= 0.95)
        mock_random.return_value = 0.6
        assert reader.random_gender() == "female"
        
        # Test "male" gender (<= 0.475)
        mock_random.return_value = 0.3
        assert reader.random_gender() == "male"

    @patch('random.random')
    def test_enrich_question(self, mock_random):
        """Test enrich_question method"""
        reader = QAndAFileReader(directory=self.test_directory)
        mock_random.return_value = 0.5  # For consistent testing
        
        # Mock the category mapping
        with patch.object(reader, 'amazon_pqa_constant_to_category_name', return_value='test_category'):
            # Mock faker methods
            reader.fake.name = Mock(return_value="John Doe")
            reader.fake.latitude = Mock(return_value="40.7128")
            reader.fake.longitude = Mock(return_value="-74.0060")
            
            enriched = reader.enrich_question('AMAZON_PQA_TEST', self.sample_question.copy())
            
            # Verify category name was added
            assert enriched['category_name'] == 'test_category'
            
            # Verify bullets were concatenated and individual bullet points removed
            expected_bullets = "Feature 1 Feature 2 Feature 3 Feature 4 Feature 5"
            assert enriched['bullets'] == expected_bullets
            for bullet in ['bullet_point1', 'bullet_point2', 'bullet_point3', 'bullet_point4', 'bullet_point5']:
                assert bullet not in enriched
            
            # Verify answers were enriched
            assert len(enriched['answers']) == 2
            for answer in enriched['answers']:
                assert 'name' in answer
                assert 'user_lat' in answer
                assert 'user_lon' in answer
                assert 'gender' in answer
                assert 'age' in answer
                assert 'product_rating' in answer
                assert isinstance(answer['user_lat'], float)
                assert isinstance(answer['user_lon'], float)
                assert answer['gender'] in ['male', 'female', 'other']
                assert 0 <= answer['age'] <= 100
                assert 1 <= answer['product_rating'] <= 5

    @patch('builtins.open', new_callable=mock_open)
    @patch('data_process.qanda_file_reader.logging')
    def test_questions_for_category_not_enriched(self, mock_logging, mock_file):
        """Test questions_for_category method without enrichment"""
        # Setup mock file content
        test_data = [
            json.dumps({"question_id": "1", "question": "Test 1"}),
            json.dumps({"question_id": "2", "question": "Test 2"}),
            json.dumps({"question_id": "3", "question": "Test 3"})
        ]
        mock_file.return_value.__iter__ = Mock(return_value=iter(test_data))
        
        reader = QAndAFileReader(directory=self.test_directory, max_number_of_docs=2)
        
        questions = list(reader.questions_for_category('AMAZON_PQA_JEANS', enriched=False))
        
        # Should only get 2 questions due to max_number_of_docs
        assert len(questions) == 2
        assert questions[0]['question_id'] == '1'
        assert questions[1]['question_id'] == '2'
        
        # Verify file was opened correctly
        mock_file.assert_called_once_with(f"{self.test_directory}/amazon_pqa_jeans.json")

    @patch('builtins.open', new_callable=mock_open)
    @patch('data_process.qanda_file_reader.logging')
    def test_questions_for_category_enriched(self, mock_logging, mock_file):
        """Test questions_for_category method with enrichment"""
        # Setup mock file content
        test_data = [json.dumps(self.sample_question)]
        mock_file.return_value.__iter__ = Mock(return_value=iter(test_data))
        
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock enrich_question method
        with patch.object(reader, 'enrich_question') as mock_enrich:
            mock_enrich.return_value = {"enriched": True}
            
            questions = list(reader.questions_for_category('AMAZON_PQA_JEANS', enriched=True))
            
            assert len(questions) == 1
            assert questions[0] == {"enriched": True}
            mock_enrich.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    def test_questions_for_category_unlimited_docs(self, mock_file):
        """Test questions_for_category method with unlimited documents"""
        # Setup mock file content with more data
        test_data = [json.dumps({"question_id": str(i)}) for i in range(5)]
        mock_file.return_value.__iter__ = Mock(return_value=iter(test_data))
        
        reader = QAndAFileReader(directory=self.test_directory, max_number_of_docs=-1)
        
        questions = list(reader.questions_for_category('AMAZON_PQA_JEANS', enriched=False))
        
        # Should get all 5 questions
        assert len(questions) == 5

    def test_questions_for_all_categories(self):
        """Test questions_for_all_categories method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock amazon_pqa_constants and questions_for_category
        reader.amazon_pqa_constants = ['AMAZON_PQA_JEANS', 'AMAZON_PQA_MONITORS']
        
        with patch.object(reader, 'questions_for_category') as mock_questions:
            mock_questions.side_effect = [
                [{"category": "jeans", "id": 1}, {"category": "jeans", "id": 2}],
                [{"category": "monitors", "id": 3}]
            ]
            
            all_questions = list(reader.questions_for_all_categories(enriched=False))
            
            assert len(all_questions) == 3
            assert all_questions[0]["category"] == "jeans"
            assert all_questions[2]["category"] == "monitors"
            
            # Verify questions_for_category was called for each constant
            assert mock_questions.call_count == 2

    def test_print_categories(self):
        """Test print_categories method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock the AMAZON_PQA_CATEGORY_MAP
        with patch.object(reader, 'AMAZON_PQA_CATEGORY_MAP', {
            'jeans': 'AMAZON_PQA_JEANS',
            'monitors': 'AMAZON_PQA_MONITORS'
        }):
            with patch('builtins.print') as mock_print:
                reader.print_categories()
                
                # Verify print was called for each category
                mock_print.assert_any_call('jeans')
                mock_print.assert_any_call('monitors')

    def test_is_category_name(self):
        """Test is_category_name method"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock the AMAZON_PQA_CATEGORY_MAP
        with patch.object(reader, 'AMAZON_PQA_CATEGORY_MAP', {
            'jeans': 'AMAZON_PQA_JEANS',
            'monitors': 'AMAZON_PQA_MONITORS'
        }):
            assert reader.is_category_name('jeans') is True
            assert reader.is_category_name('monitors') is True
            assert reader.is_category_name('nonexistent') is False

    def test_constants_and_mappings_exist(self):
        """Test that required constants and mappings exist"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Verify that the class has the required constants
        assert hasattr(reader, 'AMAZON_PQA_FILENAME_MAP')
        assert isinstance(reader.AMAZON_PQA_FILENAME_MAP, dict)
        assert len(reader.AMAZON_PQA_FILENAME_MAP) > 0
        
        # Verify some known mappings exist
        assert 'AMAZON_PQA_JEANS' in reader.AMAZON_PQA_FILENAME_MAP
        assert reader.AMAZON_PQA_FILENAME_MAP['AMAZON_PQA_JEANS'] == 'amazon_pqa_jeans.json'

    def test_category_map_consistency(self):
        """Test that AMAZON_PQA_CATEGORY_MAP is consistent with filename map"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Check if AMAZON_PQA_CATEGORY_MAP exists (it should be defined in the class)
        if hasattr(reader, 'AMAZON_PQA_CATEGORY_MAP'):
            # Verify that all constants in category map exist in filename map
            for category_name, constant in reader.AMAZON_PQA_CATEGORY_MAP.items():
                assert constant in reader.AMAZON_PQA_FILENAME_MAP, f"Constant {constant} not found in filename map"

    @patch('random.random')
    def test_enrich_question_edge_cases(self, mock_random):
        """Test enrich_question method with edge cases"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Test with empty answers
        question_no_answers = self.sample_question.copy()
        question_no_answers['answers'] = []
        
        with patch.object(reader, 'amazon_pqa_constant_to_category_name', return_value='test_category'):
            enriched = reader.enrich_question('AMAZON_PQA_TEST', question_no_answers)
            assert enriched['answers'] == []
            assert 'bullets' in enriched
            assert 'category_name' in enriched

    def test_file_size_with_missing_constant(self):
        """Test file_size method when constant is not found"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Mock category mapping that returns None for constant lookup
        with patch.object(reader, 'amazon_pqa_category_name_to_constant', return_value=None):
            with pytest.raises(Exception, match="Unknown category name test_category"):
                reader.file_size('test_category')

    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_questions_for_category_file_not_found(self, mock_file):
        """Test questions_for_category method when file is not found"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        with pytest.raises(FileNotFoundError):
            list(reader.questions_for_category('AMAZON_PQA_JEANS', enriched=False))

    def test_faker_integration(self):
        """Test that Faker is properly integrated"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Verify faker instance exists and has expected methods
        assert hasattr(reader, 'fake')
        assert hasattr(reader.fake, 'name')
        assert hasattr(reader.fake, 'latitude')
        assert hasattr(reader.fake, 'longitude')

    def test_sets_initialization(self):
        """Test that sets are properly initialized"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Verify sets are initialized as empty sets
        assert isinstance(reader.asins, set)
        assert isinstance(reader.questions, set)
        assert len(reader.asins) == 0
        assert len(reader.questions) == 0

    @patch('random.random')
    def test_random_gender_boundary_conditions(self, mock_random):
        """Test random_gender method at boundary conditions"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        # Test exact boundary values
        mock_random.return_value = 0.95
        assert reader.random_gender() == "female"  # Exactly 0.95 should be female
        
        mock_random.return_value = 0.475
        assert reader.random_gender() == "male"  # Exactly 0.475 should be male (not > 0.475)
        
        mock_random.return_value = 0.951
        assert reader.random_gender() == "other"  # Just above 0.95 should be other
        
        mock_random.return_value = 0.474
        assert reader.random_gender() == "male"  # Just below 0.475 should be male

    def test_enrich_question_product_rating_range(self):
        """Test that product rating is within expected range"""
        reader = QAndAFileReader(directory=self.test_directory)
        
        with patch.object(reader, 'amazon_pqa_constant_to_category_name', return_value='test_category'):
            # Mock faker methods
            reader.fake.name = Mock(return_value="Test Name")
            reader.fake.latitude = Mock(return_value="0.0")
            reader.fake.longitude = Mock(return_value="0.0")
            
            # Test multiple random values to ensure rating is always 1-5
            with patch('random.random') as mock_random:
                for test_value in [0.0, 0.2, 0.5, 0.8, 0.99]:
                    mock_random.return_value = test_value
                    enriched = reader.enrich_question('AMAZON_PQA_TEST', self.sample_question.copy())
                    
                    for answer in enriched['answers']:
                        assert 1 <= answer['product_rating'] <= 5
                        assert isinstance(answer['product_rating'], int)

    def test_questions_for_category_json_parsing_error(self):
        """Test questions_for_category method with invalid JSON"""
        with patch('builtins.open', mock_open(read_data="invalid json\n{\"valid\": \"json\"}")):
            reader = QAndAFileReader(directory=self.test_directory)
            
            # Should raise JSONDecodeError for invalid JSON
            with pytest.raises(json.JSONDecodeError):
                list(reader.questions_for_category('AMAZON_PQA_JEANS', enriched=False))
