#include <systemc.h>
#include <iostream>
#include <vector>
#include <cfloat>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;
typedef vector<vector<vector<float>>> Pic;
typedef vector<Pic> Tensor;
typedef vector<vector<float>> FCTensor;
typedef vector<vector<vector<vector<float>>>> WeightConv;
typedef vector<float> WeightConvBias;
typedef vector<vector<float>> WeightFC;
typedef vector<float> WeightFCBias;

SC_MODULE(AlexNet)
{
    // read information
    vector<string> ReadLabel(const string &file_name);
    Pic ReadPic(const string &file_name, int c, int h, int w);
    WeightFCBias ReadFCBiasWeight(const string file_path, int ch_out);
    WeightFC ReadFCWeight(const string file_path, int ch_in, int ch_out);
    WeightConvBias ReadConvBiasWeight(const string file_path, int ch_out);
    WeightConv ReadConvWeight(const string file_path, int kernel_x, int kernel_y, int ch_in, int ch_out);

    // AlexNet
    Tensor ApplyPadding(Tensor & input, int padding);
    Tensor Conv(WeightConv & weight_conv, WeightConvBias & weight_bias, Tensor & input, int kernel_x, int kernel_y, int ch_in, int ch_out, int stride, int padding);
    Tensor ReLU_conv(const Tensor &input);
    Tensor MaxPooling(const Tensor &input, int pool_size_x, int pool_size_y, int stride);
    Tensor AdaptiveAvgPool(const Tensor &input, int output_height, int output_width);
    FCTensor Flatten(const Tensor &input);
    FCTensor FullyConnected(const FCTensor &input, const WeightFC &weights, const WeightFCBias &bias);
    FCTensor ReLU_FC(const FCTensor &input);

    // print AlexNet parameter
    void printFCTensorSize(const FCTensor &input, const string &name);
    void printTensorSize(const Tensor &input, const string &name);

    // inference
    vector<float> softmax(const vector<float> &logits);
    void printResult(FCTensor result, vector<string> label, vector<string> pic_name);
    FCTensor inference(Tensor & input_pic);

    void process();

    vector<vector<vector<float>>> in_data;
    vector<vector<float>> out_data;
    const int PIC_NUM = 2;

    SC_CTOR(AlexNet)
    {
        SC_THREAD(process);
    }
};