#include "AlexNet.h"

using namespace std;

WeightConv AlexNet::ReadConvWeight(const string file_path, int kernel_x,
                                   int kernel_y, int ch_in, int ch_out) {
    WeightConv weight(ch_out,
                      vector<vector<vector<float>>>(
                          ch_in, vector<vector<float>>(
                                     kernel_y, vector<float>(kernel_x, 0.0))));

    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return weight;
    }

    for (int out_channel = 0; out_channel < ch_out; ++out_channel) {
        for (int in_channel = 0; in_channel < ch_in; ++in_channel) {
            for (int y = 0; y < kernel_y; ++y) {
                for (int x = 0; x < kernel_x; ++x) {
                    if (!(file >> weight[out_channel][in_channel][y][x])) {
                        cerr << "Error reading weight for out_channel="
                             << out_channel << ", in_channel=" << in_channel
                             << ", y=" << y << ", x=" << x << endl;
                        return weight;
                    }
                }
            }
        }
    }

    file.close();
    return weight;
}

WeightConvBias AlexNet::ReadConvBiasWeight(const string file_path, int ch_out) {
    WeightConvBias weight(ch_out, 0.0);
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return weight;
    }
    for (int out_channel = 0; out_channel < ch_out; ++out_channel) {
        if (!(file >> weight[out_channel])) {
            cerr << "Error reading weight for out_channel=" << out_channel
                 << endl;
            return weight;
        }
    }
    file.close();
    return weight;
}

WeightFC AlexNet::ReadFCWeight(const string file_path, int ch_in, int ch_out) {
    WeightFC weight(ch_out, vector<float>(ch_in, 0.0));

    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return weight;
    }

    for (int out_channel = 0; out_channel < ch_out; ++out_channel) {
        for (int in_channel = 0; in_channel < ch_in; ++in_channel) {
            if (!(file >> weight[out_channel][in_channel])) {
                cerr << "Error reading weight for out_channel=" << out_channel
                     << ", in_channel=" << in_channel << endl;
                return weight;
            }
        }
    }
    file.close();
    return weight;
}

WeightFCBias AlexNet::ReadFCBiasWeight(const string file_path, int ch_out) {
    WeightFCBias weight(ch_out, 0.0);
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return weight;
    }
    for (int out_channel = 0; out_channel < ch_out; ++out_channel) {
        if (!(file >> weight[out_channel])) {
            cerr << "Error reading weight for out_channel=" << out_channel
                 << endl;
            return weight;
        }
    }
    file.close();
    return weight;
}
