class TfConfig:
    OUTDIR = './model_trained'
    ## Location of training CSV file
    TRAINING_DATA_FILE = "samples/housing/stockholm-housingprices.csv"
    PREDICT_INPUT_FILE = "samples/housing/predict_input.json"
    LABEL_NAME = "final_price"
    NUMERICAL_FEATURE_COLUMNS = ["num_of_rooms","size","initial_price"]
    ## The categorical features as an array of tuples containing feature name and hash bucket size
    ## the size of the hash bucket should be at least equal to the number of expected categories
    CATEGORICAL_FEATURE_COLUMNS = [("street_name",800),("location",80),("sold_month",24)]
    ## A divider to define dimension of the categorical features in relation to bucket size. 
    ## This allows for dimensionality reduction of the categories. A value of 'n' means a dimension
    ## equals the bucket size divided by 'n'
    CATEGORICAL_FEATURE_DIMENSION_DIVIDER = 1
    ## Categorical features that have bucketized columns. Defined as an array of tuples in the format:
    ## (feature_name, min_value, max_value, step). For example, ("my_feature", 0, 11, 4) will define 
    ## the category buckets [0-3], [4-7], [8-11] - so all values falling in the same bucket are treated
    ## as the same category.
    CATEGORICAL_FEATURE_BUCKETIZED_COLUMNS = [("year_built",1700, 2020, 10),("sold_year",2010, 2020, 2),("floor_num",0, 11, 1)]
    ## A filter to apply on the source data. Rows not matching the filter will not be included in the 
    ## training dataset. Must be defined as dictionary where the keys are column names, and values their
    ## desired values (more than one value will be considered with an 'OR' operator, i.e. either one of
    ## the values are accepted)
    FEATURE_COLUMN_FILTER = {
        "type": ["Lägenhet"]
    }
    ## A dictionary of multipliers for numerical features 
    ## i.e. values for each feature column (defined as dict
    ## keys) will be multiplied by the specified value. Use
    ## fractions to perform a division.
    TRANSFORM_DATA = {
        "size": 1/10,
        "initial_price": 1/1000000,
        "final_price": 1/1000000
    }
    BATCH_SIZE = 64
    NUM_OF_TRAINING_STEPS = 10000
    DNN_CONFIG = {
        "model": "Regressor",
        "optimizer": "Ftrl",
        "hidden_units": [16,24,8],
        "Ftrl" : {
            "optimizer_learning_rate": 0.1,
            "optimizer_learning_rate_power": -0.5,
            "optimizer_initial_accumulator_value": 0.1,
            "optimizer_l1_regularization_strength": 0.03,
            "optimizer_l2_regularization_strength": 0.05,
            "l2_shrinkage_regularization_strength": 0.01,
        },
        "Adam" : {
            "optimizer_learning_rate": 0.7
        },
        "RMSprop" : {
            "optimizer_learning_rate": 0.005,
            "rho": 0.9,
            "momentum": 0.5,
            "epsilon": 1e-07,
            "centered": True
        },
        "Adagrad" : {
            "optimizer_learning_rate": 0.005,
            "epsilon": 1e-07
        }
    }
