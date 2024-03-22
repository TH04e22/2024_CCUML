import time
import numpy as np
import argparse
from Perceptron import LinearPerceptron
import os
import matplotlib.pyplot as plt

def pocket(perceptron: LinearPerceptron) -> np.ndarray:
    """
    Do the Pocket algorithm here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  
    
    """
    data, label = np.array(perceptron.data), np.array(perceptron.label)
    weight_matrix = data[0].copy()
    best_weight_matrix = weight_matrix.copy()
    iterCnt = 0
    errorCnt = 0
    dataSize = np.size(data, 0)
    lowestErrorCnt = dataSize
    
    start_time = time.process_time()
    try:
        while True:
            i = iterCnt % dataSize
            predict = np.sign(np.inner(weight_matrix, data[i]))
            if predict != label[i]:
                weight_matrix += label[i] * data[i]
            
            errorCnt = 0
            for j in range(dataSize):
                predict = np.sign(np.inner(weight_matrix, data[j]))
                if predict != label[j]:
                    errorCnt += 1
            
            if errorCnt < lowestErrorCnt:
                best_weight_matrix = weight_matrix.copy()
                lowestErrorCnt = errorCnt

            if lowestErrorCnt == 0:
                end_time = time.process_time()
                print(f'Iteration: {iterCnt}, Accurracy: {1-(lowestErrorCnt/dataSize):.3f}')
                print(f'Execution Time: {(end_time-start_time)}')
                return best_weight_matrix

            if iterCnt % 100 == 0:
                print(f'Iteration: {iterCnt}, Lowest ErrorCount: {lowestErrorCnt},Accurracy: {1-(lowestErrorCnt/dataSize):.3f}')
            iterCnt += 1
    except KeyboardInterrupt:
        end_time = time.process_time()
        print(f'Iteration: {iterCnt}, Accurracy: {1-(lowestErrorCnt/dataSize):.3f}')
        print(f'Execution Time: {(end_time-start_time)}')
        return best_weight_matrix

def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    perceptron = LinearPerceptron(args.path)
    updated_weight = pocket(perceptron=perceptron)
    data, label = np.array(perceptron.data), np.array(perceptron.label)
    dataSize = np.size(data, 0)
    errorCnt = 0
    for j in range(dataSize):
        predict = np.sign(np.inner(updated_weight, data[j]))
        if predict != label[j]:
            errorCnt += 1
    print(f'ErrorCount: {errorCnt}')

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
