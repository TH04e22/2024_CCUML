import time
import numpy as np
import argparse
import os
from Perceptron import LinearPerceptron
import matplotlib.pyplot as plt

def PLA(perceptron: LinearPerceptron) -> np.ndarray:
    """
    Do the PLA here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  

    """
           
    data, label = np.array(perceptron.data), np.array(perceptron.label)
    weight_matrix = data[0]
    epoch = 0
    errorCnt = 1

    start_time = time.process_time()
    while errorCnt != 0:
        errorCnt = 0
        for i in range(np.size(data, 0)):
            predict = np.sign(np.inner(weight_matrix, data[i]))
            if predict != label[i]:
                weight_matrix += label[i] * data[i]
                errorCnt += 1

        epoch += 1
        acc = 1-errorCnt/np.size(data, 0)
        print(f'Iteration: {epoch*np.size(data, 0)}, Accurracy: {acc:.3f}')
    end_time = time.process_time()

    print(f'Execution Time: {(end_time-start_time)}')
    return weight_matrix

def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    perceptron = LinearPerceptron(args.path)
    updated_weight = PLA(perceptron=perceptron)

    fig = plt.figure()
    axes = fig.add_axes([0.15,0.1,0.75,0.85])
    axes.set_xlim(-1100, 1100)
    axes.set_ylim(-1100, 1100)
    axes.scatter(perceptron.cor_x_pos, perceptron.cor_y_pos, c='orange', alpha=0.5, label='positive')
    axes.scatter(perceptron.cor_x_neg, perceptron.cor_y_neg, c='skyblue', alpha=0.7, label='negative')

    readLine = np.poly1d([3,5])
    x_axis = np.linspace(-1100, 1100)
    y_axis = readLine(x_axis)
    axes.plot(x_axis, y_axis, c='gray', label='y=3x+5')

    plaLine = np.poly1d([-(updated_weight[1]/updated_weight[2]), -(updated_weight[0]/updated_weight[2])])
    y_axis = plaLine(x_axis)
    axes.plot(x_axis, y_axis, c='red', alpha=0.3, label='update_weight')

    plt.xlabel('x')
    plt.ylabel('y')
    axes.legend(loc='lower left')
    plt.show()


    
if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)