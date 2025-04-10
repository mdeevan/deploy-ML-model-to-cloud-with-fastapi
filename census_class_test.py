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
    model = None
    lb = None
    df = None

    X_train = None
    X_test = None
    y_train = None
    y_test = None
    n_estimators = None
    train = None
    test = None
    preds = None

    logging.basicConfig(
        filename='logs/census.log',
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

            TestCensus.train = TestCensus.census_obj.train
            TestCensus.test = TestCensus.census_obj.test

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
            TestCensus.X_train, TestCensus.y_train, TestCensus.encoder, TestCensus.lb = TestCensus.census_obj._process_data(training_flag=True)

            logging.info(f"test_process_data : X_train shape {TestCensus.X_train.shape}, y_train shape {TestCensus.y_train.shape}")
            logging.info("process data - Train: SUCCESS")

        except AssertionError as err:
            logging.error("process data - Train: FAILURE")
            raise err

        try:
            TestCensus.X_test, TestCensus.y_test, _, _ = TestCensus.census_obj._process_data(training_flag=False, features=TestCensus.test , encoder=TestCensus.encoder, lb=TestCensus.lb)
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

            TestCensus.model = TestCensus.census_obj.model

            if TestCensus.census_obj.model is not None:
                logging.info("Training model : SUCCESS")
            else:
                logging.info("Training model : FAILURE")

        except (AssertionError, AttributeError) as err:
            logging.error("Training model: FAILURE")
            raise err

    def test_save_data_split(self, tmp_path):
        try:
            TestCensus.census_obj._save_data_split()
            logging.info('Test save selfTest save data split: SUCCESS')
        except:
            logging.info('Test save data split: FAILURE')


    def test_save_model(self, tmp_path):
        try:
            TestCensus.census_obj._save_model(tmp_path)
            logging.info('Saving Model: SUCCESS')

        except AssertionError as err:
            logging.error("Saving Model: FAILURE")
            raise err

    def test_save_model_negative(self, tmp_path):
        try:
            os.mkdir("testing", 0o444)
            TestCensus.census_obj._save_model("testing")
            logging.info('Saving Model: SUCCESS')
            os.rmdir("testing")

        except (PermissionError, AssertionError) as err:
            os.rmdir("testing")
            logging.error("Saving Model: FAILURE TEST success")
            # raise err
            pass

    def test_make_inference(self, tmp_path):
        try:
            # print(f"X_Test : {TestCensus.X_test}")
            TestCensus.preds = TestCensus.census_obj.make_inference(TestCensus.model, TestCensus.X_test, path=tmp_path)

            logging.error("Make inference : SUCCESS")

        except:
            logging.error("Make inference : FAILED")


    def test_compute_metrics(self, tmp_path):
        try:
            TestCensus.census_obj._compute_metrics(TestCensus.y_test, TestCensus.preds, path=tmp_path)
            logging.error("Metrics compute : SUCCESS")

        except:
            logging.error("Metrics compute : FAILED")
            

    def test_execute_training(self):

        TestCensus.census_obj.execute_training()

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