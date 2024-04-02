#include "AlexNet.h"

using namespace std;

void AlexNet::process() {
    Tensor input_pic(
        PIC_NUM, vector<vector<vector<float>>>(
                     3, vector<vector<float>>(224, vector<float>(224, 0.0))));

    for (int b = 0; b < PIC_NUM; b++)
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 224; h++)
                for (int w = 0; w < 224; w++)
                    input_pic[b][c][h][w] = input_data[b][c][h][w];
    vector<vector<float>> output = inference(input_pic);
    for (int p = 0; p < PIC_NUM; p++) {
        for (int i = 0; i < 1000; i++) {
            output_data[p][i] = output[p][i];
        }
    }
}