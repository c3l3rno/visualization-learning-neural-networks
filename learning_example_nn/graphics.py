import matplotlib.pyplot as plt

# Imports from own programs
from gen_learn_data import generate_data
from learning_example_nn.inference import get_values_of_model

def make_img(name, epoch=''):
    #get data
    data_train = generate_data(10)
    data_model = get_values_of_model(75)

    # Extrahiere die x und y-Werte aus der Liste
    x_values_train = [point[0] for point in data_train]
    y_values_train = [point[1] for point in data_train]

    x_values_model = [point[0] for point in data_model]
    y_values_model = [point[1] for point in data_model]
    
    #configure graph
    plt.figure(figsize=(8, 6)) #800x600

    plt.ylim(-0.2, 1) #set range of y-values
    ax = plt.gca()  #select axis
    ax.spines['bottom'].set_position(('data', 0))  #x-axis on same height as y=0

    # Blue: training data
    plt.plot(x_values_train, y_values_train, marker='o', linestyle='', color='b', markersize=1)

    # Red: model output
    plt.plot(x_values_model, y_values_model, marker='o', linestyle='-', color='r', markersize=1)


    #add some lables
    plt.title(f"Epoch {epoch}")
    plt.xlabel('x-values')
    plt.ylabel('y-values')

    #save to an image folder
    plt.savefig(f'learning_example_nn\\images\\{name}.png') 
    
    #close to free memory
    plt.close()