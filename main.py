from model_trainer import ModelTrainer
#from samples.housing.tfconfig import TfConfig as cfg
from samples.wine.tfconfig import TfConfig as cfg

if __name__ == '__main__':
    trainer = ModelTrainer(cfg)
    trainer.train_dnn(start_fresh = True)
    trainer.evaluate_dnn()
    trainer.predict()