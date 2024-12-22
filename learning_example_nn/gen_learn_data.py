#generates the data to train the model
def generate_data(detail):
    #inclusive range (ds stands for definition set)
    ds = (-10, 10)
    data = []
    x = ds[0]
    its = (abs(ds[0]) + abs(ds[1]))*detail
    for i in range(its):
        #the function that can be changed
        y = x*x+2 - x*x*x*0.05

        data.append([x,y])
        x += 1/detail


    return(normailze_data(data))

#bringing the values into a range of -1 to 1
def normailze_data(data):
    x = [point[0] for point in data]
    y = [point[1] for point in data]

    modf_x = 1/max(x)
    modf_y = 1/max(y)
    
    #change data
    normalized_x = [xi * modf_x for xi in x]
    normalized_y = [yi * modf_y for yi in y]
    
    #return it as a list (2d)
    return list(zip(normalized_x, normalized_y))