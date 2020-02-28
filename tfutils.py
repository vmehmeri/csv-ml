import tensorflow as tf

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def pandas_train_input_fn(df, batch_size, feature_names, label_name):
    #1. Convert dataframe into correct (features,label) format for Estimator API
    print("[TRAIN] Converting dataframe into correct (features,label) format for Estimator API")
    dataset = tf.data.Dataset.from_tensor_slices(tensors = (dict(df[feature_names]), df[label_name]))
    
    # Note:
    # If we returned now, the Dataset would iterate over the data once  
    # in a fixed order, and only produce a single element at a time.
    
    #2. Shuffle, repeat, and batch the examples.
    print("[TRAIN] Shuffling and batching dataset")
    dataset = dataset.shuffle(buffer_size = 1000).repeat(count = None).batch(batch_size = batch_size)

    print(bcolors.OKGREEN, "\n[TRAIN] dataset is ready\n", bcolors.ENDC)
    return dataset

def pandas_eval_input_fn(df, batch_size, feature_names, label_name):
    #1. Convert dataframe into correct (features,label) format for Estimator API
    print("\n[EVAL] Converting dataframe into correct (features,label) format\n")
    dataset = tf.data.Dataset.from_tensor_slices(tensors = (dict(df[feature_names]), df[label_name]))
    
    #2.Batch the examples.
    print("[EVAL] Batching the examples\n")
    dataset = dataset.batch(batch_size = batch_size)
   
    return dataset

def convert_month_to_str(month_digit):
    month_dict = {
        1 : "jan",
        2 : "feb",
        3 : "mar",
        4 : "apr",
        5 : "maj",
        6 : "jun",
        7 : "jul",
        8 : "aug",
        9 : "sep",
        10 : "okt",
        11 : "nov",
        12 : "dec",
    }

    return month_dict[month_digit]