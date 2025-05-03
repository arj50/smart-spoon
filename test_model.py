from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('food_model.h5')

# Print model summary to verify it's loaded correctly
model.summary()

