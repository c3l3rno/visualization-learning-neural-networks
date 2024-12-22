#way better performance with batches (something around 3 to 20x)
from tensorflow import keras
from tensorflow import range, expand_dims, stack, squeeze #only import the things we need to improve performance

model = None
model_path = 'graph.keras'

def inference(input_tensor):
    prediction = model(input_tensor, training=False)
    return prediction


def get_values_of_model(detail):
    #reload on every call
    global model
    model = keras.models.load_model(model_path)

    x_values = range(-1.0, 1.0, delta=2.0/(detail*2)) #generate the input data
    input_tensor = expand_dims(x_values, axis=1)

    predictions = inference(input_tensor) #make some predictions

    eval_data = stack([x_values, squeeze(predictions)], axis=1).numpy() #convert data to readable data

    return eval_data.tolist() #convert data to list
