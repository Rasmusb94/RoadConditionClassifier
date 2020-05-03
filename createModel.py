#Create a new model
from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
#Select Algorithm
model_trainer.setModelTypeAsDenseNet()
model_trainer.setDataDirectory("data")
#Num_objects = number of categories
model_trainer.trainModel(num_objects=4, num_experiments=200, enhance_data=True, batch_size=16, show_network_summary=True)