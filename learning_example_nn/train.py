def train_model(epochs):
    #everything fine with imports
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    #own imports
    from gen_learn_data import generate_data

    #generate some learning data: 1/10 increment
    data = generate_data(10)

    #extract data
    x_train = [point[0] for point in data]
    y_train = [point[1] for point in data]

    #create model 25x25x1
    model = Sequential([
        Dense(25, activation='relu', input_shape=(1,)),
        Dense(25, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')


    #train; verbose means no terminal information
    model.fit(x_train, y_train, epochs=epochs, batch_size=10, verbose=0)

    #save it 
    model.save(f'learning_example_nn\\graph.keras')