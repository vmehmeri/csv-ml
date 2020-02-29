from tfutils import *
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
import shutil
import sys
import os
import json

tf.get_logger().setLevel('INFO')

class ModelTrainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.df = pd.read_csv(self.cfg.TRAINING_DATA_FILE)
        self.df.dropna(inplace=True)

        # Random state ensures a fixed random seed
        self.df, self.df_eval = train_test_split(self.df, test_size=0.2, random_state=10) #shuffle = False, stratify = None)

        for feat_col, feat_vals in self.cfg.FEATURE_COLUMN_FILTER.items():
            self.df, self.df_eval = self.df[self.df[feat_col].isin(feat_vals)], self.df_eval[self.df_eval[feat_col].isin(feat_vals)]

        self.featcols = []
        self.feature_names = []
        self.label_name = self.cfg.LABEL_NAME
        self.model = None

        for feat, multiplier in self.cfg.TRANSFORM_DATA.items():
            self.df[feat] *= multiplier
            self.df_eval[feat] *= multiplier

        for num_col in self.cfg.NUMERICAL_FEATURE_COLUMNS:
            self.featcols.append(tf.feature_column.numeric_column(num_col))
            self.feature_names.append(num_col)

        for cat_col, cat_hash_bucket_size in self.cfg.CATEGORICAL_FEATURE_COLUMNS:
            self.featcols.append(
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_hash_bucket(cat_col,cat_hash_bucket_size), dimension=cat_hash_bucket_size//self.cfg.CATEOGRICAL_FEATURE_DIMENSION_DIVIDER
                    )
            )
            self.feature_names.append(cat_col)

        # for cat_col_vocab, cat_vobab_list in self.cfg.CATEGORICAL_FEATURE_COLUMNS_WITH_VOCAB:
        #     self.featcols.append(tf.feature_column.categorical_column_with_vocabulary_list(cat_col_vocab,cat_vobab_list))
        #     self.feature_names.append(cat_col_vocab)

        for cat_col_buck, lower_bound, upper_bound, step in self.cfg.CATEGORICAL_FEATURE_BUCKETIZED_COLUMNS:
            self.featcols.append(tf.feature_column.bucketized_column(tf.feature_column.numeric_column(cat_col_buck), boundaries= np.arange(lower_bound, upper_bound, step).tolist()))
            self.feature_names.append(cat_col_buck)

    @staticmethod
    def transform_input_feature(feature_dict, data_transform_dict):
        for feature_name, multiplier in data_transform_dict.items():
            if feature_name in feature_dict:
                for idx in range(0,len(feature_dict[feature_name])):
                    feature_dict[feature_name][idx] *= multiplier


    @staticmethod
    def get_rmse(model, df, batch_size, feature_names, label_name, data_transform_dict):
        metrics = model.evaluate(input_fn = lambda: pandas_eval_input_fn(df, batch_size, feature_names, label_name))
        avg_loss = metrics["average_loss"]**.5
        label_mean = df[label_name].mean()

        if label_name in data_transform_dict:
            #avg_loss = int(denormalize_val(self.cfg.LABEL_NAME,avg_loss))
            avg_loss = int(avg_loss * 1/data_transform_dict[label_name])
            label_mean = label_mean * 1/data_transform_dict[label_name]

        avg_loss_percentage = round((avg_loss/label_mean)*100,2)
        return avg_loss, avg_loss_percentage

    @staticmethod
    def print_rmse(model, df, batch_size, feature_names, label_name, data_transform_dict):
        avg_loss, avg_loss_percentage = ModelTrainer.get_rmse(model, df, batch_size, feature_names, label_name, data_transform_dict)
        print(bcolors.BOLD, f"\nRMSE on dataset = {avg_loss} ({avg_loss_percentage}% of average label value)", bcolors.ENDC)


    #####################################################################################
    #######################                                 #############################
    #######################        Deep Neural Network      #############################
    #######################                                 #############################
    #####################################################################################
    def train_dnn(self, start_fresh=False):
        print("Training DNN...")

        if start_fresh:
            shutil.rmtree(path = self.cfg.OUTDIR, ignore_errors = True) 

        if self.cfg.DNN_CONFIG['optimizer'] == "Ftrl":
            self.model = tf.estimator.DNNRegressor(
                hidden_units = self.cfg.DNN_CONFIG['hidden_units'], # specify neural architecture,
                optimizer=tf.optimizers.Ftrl(
                    learning_rate=self.cfg.DNN_CONFIG['Ftrl']['optimizer_learning_rate'],
                    learning_rate_power=self.cfg.DNN_CONFIG['Ftrl']['optimizer_learning_rate_power'],
                    initial_accumulator_value=self.cfg.DNN_CONFIG['Ftrl']['optimizer_initial_accumulator_value'],
                    l1_regularization_strength=self.cfg.DNN_CONFIG['Ftrl']['optimizer_l1_regularization_strength'],
                    l2_regularization_strength=self.cfg.DNN_CONFIG['Ftrl']['optimizer_l2_regularization_strength'],
                    l2_shrinkage_regularization_strength=self.cfg.DNN_CONFIG['Ftrl']['l2_shrinkage_regularization_strength'],
                ),
                activation_fn=tf.nn.relu,
                feature_columns = self.featcols, 
                model_dir = self.cfg.OUTDIR,
                config = tf.estimator.RunConfig(tf_random_seed = 1)
            )
        elif self.cfg.DNN_CONFIG['optimizer'] == "Adam":
            # Change estimator type
            self.model = tf.estimator.DNNRegressor(
                feature_columns= self.featcols,
                hidden_units = self.cfg.DNN_CONFIG['hidden_units'], # specify neural architecture,
                optimizer=tf.optimizers.Adam(
                    learning_rate=self.cfg.DNN_CONFIG['Adam']['optimizer_learning_rate'],
                ),
                model_dir = self.cfg.OUTDIR)
        elif self.cfg.DNN_CONFIG['optimizer'] == "RMSprop":
            # Change estimator type
            self.model = tf.estimator.DNNRegressor(
                feature_columns= self.featcols,
                hidden_units = self.cfg.DNN_CONFIG['hidden_units'], # specify neural architecture,
                optimizer=tf.optimizers.RMSprop(
                    learning_rate=self.cfg.DNN_CONFIG['RMSprop']['optimizer_learning_rate'], 
                    rho=self.cfg.DNN_CONFIG['RMSprop']['rho'], 
                    momentum=self.cfg.DNN_CONFIG['RMSprop']['momentum'], 
                    epsilon=self.cfg.DNN_CONFIG['RMSprop']['epsilon'], 
                    centered=self.cfg.DNN_CONFIG['RMSprop']['centered'],
                    name='RMSprop'
                ),
                model_dir = self.cfg.OUTDIR)
        elif self.cfg.DNN_CONFIG['optimizer'] == "Adagrad":
            # Change estimator type
            self.model = tf.estimator.DNNRegressor(
                feature_columns= self.featcols,
                hidden_units = self.cfg.DNN_CONFIG['hidden_units'], # specify neural architecture,
                optimizer=tf.optimizers.RMSprop(
                    learning_rate=self.cfg.DNN_CONFIG['Adagrad']['optimizer_learning_rate'], 
                    epsilon=self.cfg.DNN_CONFIG['Adagrad']['epsilon'], 
                    name='Adagrad'
                ),
                model_dir = self.cfg.OUTDIR)

        else:
            raise "Unsupported optimizer"

        ## Train
        print("\n==================== TRAINING DNN REGRESSOR ========================\n")
        self.model.train(
            input_fn = lambda: pandas_train_input_fn(self.df, self.cfg.BATCH_SIZE, self.feature_names, self.label_name), 
            max_steps = self.cfg.DNN_REGRESSOR_NUM_OF_STEPS)

        

    def evaluate_dnn(self):
        ## Evaluate
        print("\n==================== EVALUATING DNN REGRESSOR ========================\n")
        ModelTrainer.print_rmse(self.model, self.df_eval, self.cfg.BATCH_SIZE, self.feature_names, self.label_name, self.cfg.TRANSFORM_DATA) 

    def _predict_input_fn(self):
        with open(self.cfg.PREDICT_INPUT_FILE) as input_json:
            features = json.load(input_json)

        ModelTrainer.transform_input_feature(features, self.cfg.TRANSFORM_DATA)
        
        return features

    def predict(self):
        ## Predict
        predictions = self.model.predict(self._predict_input_fn)

        #print("DNN Predictions:")
        num_of_predictions = len(list(self._predict_input_fn().values())[0])
        for _ in range(num_of_predictions):
            if self.cfg.LABEL_NAME in self.cfg.TRANSFORM_DATA:
                transformed_predicted_value = next(predictions)['predictions'][0]
                pred_val = int(transformed_predicted_value/self.cfg.TRANSFORM_DATA[self.cfg.LABEL_NAME])
            else:
                pred_val = next(predictions)['predictions'][0]
            
            print(bcolors.OKGREEN, "\nPredicted Value: ", pred_val, bcolors.ENDC)
            return pred_val
