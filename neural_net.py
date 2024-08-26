from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

def build_model():
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Dense(64, activation='relu', input_shape=(11,)))
    
    # Hidden layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    
    # Output layer (changed from 7 to 6)
    model.add(layers.Dense(6, activation='linear'))
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=losses.mean_squared_error,
                  metrics=[metrics.mae])
    
    return model

# Create the model
lemonade_model = build_model()

# Print model summary
lemonade_model.summary()