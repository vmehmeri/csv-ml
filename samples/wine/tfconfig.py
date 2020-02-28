class TfConfig:
    OUTDIR = './model_trained'
    ## Location of training CSV file
    TRAINING_DATA_FILE = "samples/wine/winemag-data.csv"
    PREDICT_INPUT_FILE = "samples/wine/predict_input.json"
    LABEL_NAME = "points"
    NUMERICAL_FEATURE_COLUMNS = []
    ## The categorical features as an array of tuples containing feature name and hash bucket size
    ## the size of the hash bucket should be at least equal to the number of expected categories
    CATEGORICAL_FEATURE_COLUMNS = [("country",100),("variety",300)]
    ## A divider to define embedding dimension of the categorical features in relation to bucket size. 
    ## This allows for dimensionality reduction (embedding) of sparse categories. A value of 'n' means a dimension
    ## equals the bucket size divided by 'n'. For example, n = 2 means values will be embedded into half as many categories
    CATEOGRICAL_FEATURE_DIMENSION_DIVIDER = 2
    ## Categorical features that have bucketized columns. Defined as an array of tuples in the format:
    ## (feature_name, min_value, max_value, step). For example, ("my_feature", 0, 11, 4) will define 
    ## the category buckets [0-3], [4-7], [8-11] - so all values falling in the same bucket are treated
    ## as the same category.
    CATEGORICAL_FEATURE_BUCKETIZED_COLUMNS = [("price", 0, 1000, 5)]
    ## A filter to apply on the source data. Rows not matching the filter will not be included in the 
    ## training dataset. Must be defined as dictionary where the keys are column names, and values their
    ## desired values (more than one value will be considered with an 'OR' operator, i.e. either one of
    ## the values are accepted)
    FEATURE_COLUMN_FILTER = {
    }
    ## A dictionary of multipliers for numerical features 
    ## i.e. values for each feature column (defined as dict
    ## keys) will be multiplied by the specified value. Use
    ## fractions to perform a division.
    TRANSFORM_DATA = {
        "price": 1/100,
        "points": 1/100
    }
    BATCH_SIZE = 32
    DNN_REGRESSOR_NUM_OF_STEPS = 5000
    DNN_CONFIG = {
        "optimizer": "Ftrl",
        "hidden_units": [4,64,32],
        "Ftrl" : {
            "optimizer_learning_rate": 0.3,
            "optimizer_learning_rate_power": -0.5,
            "optimizer_initial_accumulator_value": 0.1,
            "optimizer_l1_regularization_strength": 0.2,
            "optimizer_l2_regularization_strength": 0.5,
        },
        "Adam" : {
            "optimizer_learning_rate": 0.2
        }
    }
