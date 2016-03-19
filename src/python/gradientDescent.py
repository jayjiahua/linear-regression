#coding=utf-8

import os

class GradientDescent:
    def __init__(self, dimension):
        self.theta_list = [0] * dimension
        self.x_sample = []
        self.y_sample = []
        self.dimension = dimension
        self.alpha = 0.01
        self.sample_count = 0

    def load_sample(self, filename):
        with open(filename) as file:
            first_line = file.readline()
            k = 0
            for line in file.readlines():
                k += 1
                # if k > 3000:
                #     break
                line = line.strip()
                line_list = map(lambda n: float(n), line.split(','))
                self.x_sample.append([1] + line_list[1:self.dimension])
                self.y_sample.append(line_list[self.dimension])
                self.sample_count += 1

    def h_theta(self, x_i):
        return sum([x * y for x, y in zip(self.theta_list, x_i)])

    def j_theta(self):
        return sum([(self.h_theta(x) - y) ** 2 for x, y in zip(self.x_sample, self.y_sample)]) / 2

    def training(self, filename):
        self.load_sample(filename)

        last_j_theta = None
        current_j_theta = None
        training_step = 0
        last_theta_list = None

        while True:
            #print training_step, last_j_theta, current_j_theta
            if training_step > 10000:
                break
            last_theta_list = self.theta_list
            last_j_theta = current_j_theta
            training_step += 1
            for j in range(0, self.dimension):
                #print j
                temp = sum([(y - self.h_theta(x)) * x[j] for x, y in zip(self.x_sample, self.y_sample)])
                self.theta_list[j] += self.alpha * temp / float(self.sample_count)

            current_j_theta = self.j_theta()
            print current_j_theta
            if last_j_theta == None:
                last_j_theta = current_j_theta
            elif abs(current_j_theta - last_j_theta) <= 1:
                self.theta_list = last_theta_list
                print self.theta_list
                break

    def predicting(self, filename):
        test_file = open(filename)
        output_file = open("./data/outputPython", "w")
        try:
            first_line = test_file.readline()
            output_file.write("id,reference\n")
            for line in test_file.readlines():
                line = line.strip()
                line_list = map(lambda n: float(n), line.split(','))
                x_sample = [1] + line_list[1:self.dimension]
                output_file.write("{0},{1}\n".format(str(int(line_list[0])), str(sum([x * y for x, y in zip(self.theta_list, x_sample)]))))
        finally:
            test_file.close()
            output_file.close()


if __name__ == '__main__':
    gradient_descent = GradientDescent(dimension=385)
    gradient_descent.training("./data/train.csv")
    gradient_descent.predicting("./data/test.csv")
