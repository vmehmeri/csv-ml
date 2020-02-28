# CSV-ML 
## Running Machine Learning Linear Regressions on any structured CSV dataset
This is a python program that uses Tensorflow to read data from a CSV file, and run a Deep Neural Network with either Ftrl or Adam optimizers for linear regression. 

Create a tfconfig.py Python configuration file (as seen in the samples directory) to specify the configurations of your model. The example below is for the prediction of housing prices in Stockholm:

```
class TfConfig:
    OUTDIR = './model_trained'
    ## Location of training CSV file
    TRAINING_DATA_FILE = "samples/housing/stockholm-housingprices.csv"
    PREDICT_INPUT_FILE = "samples/housing/predict_input.json"
    LABEL_NAME = "final_price"
    NUMERICAL_FEATURE_COLUMNS = ["num_of_rooms","size","initial_price"]
    ## The categorical features as an array of tuples containing feature name and hash bucket size
    ## the size of the hash bucket should be at least equal to the number of expected categories
    CATEGORICAL_FEATURE_COLUMNS = [("street_name",1600),("location",160),("sold_month",16)]
    ## A divider to define dimension of the categorical features in relation to bucket size. 
    ## This allows for dimensionality reduction of the categories. A value of 'n' means a dimension
    ## equals the bucket size divided by 'n'
    CATEOGRICAL_FEATURE_DIMENSION_DIVIDER = 2
    ## Categorical features that have bucketized columns. Defined as an array of tuples in the format:
    ## (feature_name, min_value, max_value, step). For example, ("my_feature", 0, 11, 4) will define 
    ## the category buckets [0-3], [4-7], [8-11] - so all values falling in the same bucket are treated
    ## as the same category.
    CATEGORICAL_FEATURE_BUCKETIZED_COLUMNS = [("year_built",1700, 2020, 5),("sold_year",2010, 2020, 1),("floor_num",0, 11, 3)]
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
    DNN_REGRESSOR_NUM_OF_STEPS = 5000
    DNN_CONFIG = {
        "optimizer": "Ftrl",
        "hidden_units": [32,64,32],
        "Ftrl" : {
            "optimizer_learning_rate": 0.1,
            "optimizer_learning_rate_power": -0.5,
            "optimizer_initial_accumulator_value": 0.1,
            "optimizer_l1_regularization_strength": 0.1,
            "optimizer_l2_regularization_strength": 0.2,
        },
        "Adam" : {
            "optimizer_learning_rate": 0.2
        }
    }

```

Make sure TRAINING_DATA_FILE points to your CSV file, and PREDICT_INPUT_FILE points to a JSON file containing the input non-labeled values you want to use for prediction. Below is an example for a single input for this problem:

```
{
    "size": [50],
    "num_of_rooms": [2],
    "year_built": [2002],
    "floor_num": [4],
    "street_name": ["Kungsgatan"],
    "sold_year": [2020],
    "sold_month": ["jan"],
    "location": ["Stockholm"],
    "type": ["Lägenhet"],
    "initial_price": [2795000]
}
```

## Usage
To run the application, set up a virtual environment with Python 3 and install all the requirements.

` pip install -r requirements.txt `

Before running the app, edit the following line in main.py:

`from samples.wine.tfconfig import TfConfig as cfg`

with the relative path to import your tfconfig.py file. 

Then run:

` python main.py `

The RMSE and predicted value(s) will be shown in standard output.