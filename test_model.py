'''
Author: Muhammad Naveed
Created On : April 4th, 2025
'''

import os
import logging
import pytest

import census_class as cls


class TestCensus():
    '''
    declare class level properties, as each test depends upon the results from
    the previous test
    '''

    census_obj = None
    encoder = None
    lb = None
    df = None

    X_train = None
    X_test = None
    y_train = None
    y_test = None
    n_estimators = None
    

    logging.basicConfig(
        filename='./logs/census.log',
        level=logging.INFO,
        filemode='w',
        force=True,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    def setup_class(self):
        '''
        create object necessary for the test
        '''

        TestCensus.census_obj = cls.Census(nrows=50)

        TestCensus.logging = logging.getLogger()

        TestCensus.logging.info("setup_class")
        TestCensus.logging.info("class created, %s rows to read",
                                       TestCensus.census_obj.nrows)


    def teardown_class(self):
        '''
        release the object memory at the end of testing
        '''
        TestCensus.logging.info("Teardown_class")
        TestCensus.cc_obj = None
        TestCensus.logging = None


    def test_import_data(self, tmp_path):
        '''
        test data import - 
        INPUT:
            none
        OUTPUT:
            none
        '''

        try:
            TestCensus.census_obj._read_data()
            TestCensus.df = TestCensus.census_obj.data

            TestCensus.logging.info(f"Testing import_data {TestCensus.df.shape}: SUCCESS")

        except FileNotFoundError as err:
            TestCensus.logging.error("Testing import_eda: The file \
                wasn't found")
            raise err

        try:
            assert TestCensus.census_obj.data.shape[0] > 0
            assert TestCensus.census_obj.data.shape[1] > 0

        except AssertionError as err:
            TestCensus.logging.error("Testing import_data: The file \
            doesn't appear to have rows and columns")
            raise err

    def test_split_data(self, tmp_path):
        '''
        test test split data
        '''

        try:
            TestCensus.census_obj._split_data()

            logging.info("perform split data: SUCCESS")

        except (AssertionError, KeyError, ValueError) as err:
            logging.error("perform split data: FAILURE")
            raise err

        lst_obj = [TestCensus.census_obj.train,
                   TestCensus.census_obj.test]
        lst_str = ["train", "test" ]

        for i, obj in enumerate(lst_obj):
            try:
                if obj.shape[0] > 0:
                    logging.info(f"%s has a shape of {obj.shape}", lst_str[i])

            except (AssertionError, KeyError, AttributeError) as err:
                logging.error("%s failed to be created", lst_str[i])
                raise err

    def test_process_data(self):
        '''
        test process data
        INPUT:
            None
        OUTPUT:
            endcoder, and lb
        '''

        try:
            # self.X_train, self.y_train, self.encoder, self.lb = TestCensus.census_obj._process_data(True)
            TestCensus.X_train, TestCensus.y_train, TestCensus.encoder, TestCensus.lb = TestCensus.census_obj._process_data(True)
            logging.info(f"test_process_data : X_train shape {TestCensus.X_train.shape}, y_train shape {TestCensus.y_train.shape}")
            logging.info("process data - Train: SUCCESS")

        except AssertionError as err:
            logging.error("process data - Train: FAILURE")
            raise err

        try:
            X_test, y_test, _, _ = TestCensus.census_obj._process_data(False, TestCensus.encoder, TestCensus.lb)
            logging.info("process data - Test: SUCCESS")

        except AssertionError as err:
            logging.error("process data - Test: FAILURE")
            raise err

    def test_train_models(self ):
        '''
        test train_models
        INPUT:
            None
        OUTPUT:
            None

        '''

        logging.info(f"test_train_model : X_train shape {TestCensus.X_train.shape}, y_train shape {TestCensus.y_train.shape}")
        try:
            TestCensus.census_obj._train_model(TestCensus.X_train,
                                               TestCensus.y_train, 
                                               TestCensus.census_obj.n_estimators)

            if TestCensus.census_obj.model is not None:
                logging.info("Training model : SUCCESS")
            else:
                logging.info("Training model : FAILURE")

        except (AssertionError, AttributeError) as err:
            logging.error("Training model: FAILURE")
            raise err

    def test_save_model(self, tmp_path):
        try:
            TestCensus.census_obj._save_model(tmp_path)
            logging.info('Saving Model: SUCCESS')

        except AssertionError as err:
            logging.error("Saving Model: FAILURE")
            raise err


            

    # def test_save_data_split(self, tmp_path):
    #     try:
    #         TestCensus.census_obj._save_data_split(tmp_path)
    #         logging.info('Saving data split: SUCCESS')

    #     except AssertionError as err:
    #         logging.error("Saving data split: FAILURE")
    #         raise err



if __name__ == "__main__":
    pass
    # import pytest
    # pytest.main([__file__])
    # test_import_data()