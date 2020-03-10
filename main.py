from model_trainer import ModelTrainer
#from samples.titanic.tfconfig import TfConfig as cfg
from samples.wine.tfconfig import TfConfig as cfg
import sys

if __name__ == '__main__':
    trainer = ModelTrainer(cfg)
    trainer.train_dnn(start_fresh = True)
    trainer.evaluate_dnn()
    trainer.predict()