#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace std;

double xSample[25000][385];
double ySample[25000];

class GradientDescent {
private:
    double theta[385];
    double alpha;
    int sampleCount;
    int dimension;

    void loadSample() {
        ifstream inputFile("./data/train.txt");
        int temp;
        for (int i = 0 ; i < 25000 ; i++) {
            xSample[i][0] = 1;
            inputFile >> temp;
            for (int j = 1 ; j < 385 ; j++) {
                inputFile >> xSample[i][j];
            }
            inputFile >> ySample[i];
        }
    }

    double hTheta(int index) {
        double sum = 0;
        for (int i = 0 ; i < 385 ; i++) {
            sum += theta[i] * xSample[index][i];
        }
        return sum;
    }

    double jTheta() {
        double sum = 0;
        for (int i = 0 ; i < 25000 ; i++) {
            sum += (hTheta(i) - ySample[i]) * (hTheta(i) - ySample[i]);
        }
        return sum / 2;
    }

public:
    GradientDescent(int dimen) {
        dimension = dimen;
        for (int i = 0 ; i < dimension ; i++) {
            theta[i] = 0;
        }
        alpha = 0.0001;
        sampleCount = 0;
    }
    void training() {
        loadSample();

        int step = 0;

        double lastJTheta = 0;
        double currentJTheta = 0;
        double lastTheta[385];

        while (true) {
            step++;
            if (step > 10000) {
                break;
            }

            lastJTheta = currentJTheta;

            for (int i = 0 ; i < 385 ; i++) {
                lastTheta[i] = theta[i];
            }

            for (int j = 0 ; j < 385 ; j++) {
                double tempSum = 0;
                for (int i = 0 ; i < 25000 ; i++) {
                    tempSum += (ySample[i] - hTheta(i)) * xSample[i][j];
                    theta[j] += alpha * tempSum / 25000.0;
                }
            }

            currentJTheta = jTheta();
            printf("%d: %lf\n", step, currentJTheta);

            if (lastJTheta != 0 && lastJTheta <= currentJTheta) {
                for (int i = 0 ; i < 385 ; i++) {
                    theta[i] = lastTheta[i];
                    printf("%lf ", theta[i]);
                }
                break;
            }

        }
    }

    void predicting() {
        ifstream inputFile("./data/test.txt");
        ofstream outputFile("./data/outC.csv");
        outputFile << "id,reference\n";
        for (int i = 0 ; i < 25000 ; i++) {
            int id;
            double x = 0;
            double reference = theta[0];
            inputFile >> id;
            for (int j = 1 ; j < 385 ; j++) {
                inputFile >> x;
                reference += x * theta[j];
            }
            outputFile << id << "," << reference << "\n";
        }
        outputFile.close();
    }

};


int main() {
    GradientDescent gradientDescent = GradientDescent(385);
    gradientDescent.training();
    gradientDescent.predicting();
    return 0;
}
