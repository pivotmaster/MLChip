// #include "AlexNet.h"
// #include "Pattern.h"
#include "Monitor.h"
#include "clockreset.h"
#include <iostream>
using namespace std;
#include <systemc.h>
#include <iostream>
#include <vector>
#include <cfloat>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
typedef std::vector<std::vector<std::vector<float>>> Pic;
typedef std::vector<Pic> Tensor;
typedef std::vector<std::vector<float>> FCTensor;
typedef std::vector<std::vector<std::vector<std::vector<float>>>> WeightConv;
typedef std::vector<float> WeightConvBias;
typedef std::vector<std::vector<float>> WeightFC;
typedef std::vector<float> WeightFCBias;
std::vector<float> softmax(const std::vector<float> &logits)
{
    std::vector<float> exp_values;
    std::transform(logits.begin(), logits.end(), std::back_inserter(exp_values), [](float logit)
                   { return std::exp(logit); });

    float sum = std::accumulate(exp_values.begin(), exp_values.end(), 0.0f);
    std::vector<float> probabilities;
    std::transform(exp_values.begin(), exp_values.end(), std::back_inserter(probabilities), [sum](float exp_value)
                   { return exp_value / sum; });

    return probabilities;
}
void printResult(FCTensor result, std::vector<std::string> label)
{
    cout << label[287] << endl;
    for (int i = 0; i < result.size(); i++)
    {
        std::vector<float> possibility = softmax(result[i]);
        std::vector<std::tuple<int, float, float>> indexedPossibilityWithScore;
        for (int j = 0; j < possibility.size(); j++)
        {
            indexedPossibilityWithScore.push_back(std::make_tuple(j, possibility[j], result[i][j]));
        }
        std::vector<std::tuple<int, float, float>> topTen(10);
        std::partial_sort_copy(indexedPossibilityWithScore.begin(), indexedPossibilityWithScore.end(), topTen.begin(), topTen.end(),
                               [](const std::tuple<int, float, float> &left, const std::tuple<int, float, float> &right)
                               {
                                   return std::get<1>(left) > std::get<1>(right);
                               });

        std::cout << "Top 10 Results:" << std::endl;
        for (auto &[index, val, originalScore] : topTen)
        {
            std::cout << "Index: " << index << ", Label: " << label[index]
                      << ", Probability: " << val << ", Original Score: " << originalScore << std::endl;
        }
    }
}
void printFCTensorSize(const FCTensor &input, const string &name)
{
    int batch_size = input.size();
    int depth = input[0].size();

    cout << std::left << setw(5) << name << " size: ";
    cout << std::right << setw(6) << batch_size << setw(6) << depth << "\n";
}
void printTensorSize(const Tensor &input, const string &name)
{
    int batch_size = input.size();
    int depth = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    cout << std::left << setw(5) << name << " size: ";
    cout << std::right << setw(6) << batch_size << setw(6) << depth << setw(6) << input_height << setw(6) << input_width << "\n";
}
FCTensor Flatten(const Tensor &input)
{
    int batch_size = input.size();
    int c_size = input[0].size();
    int h_size = input[0][0].size();
    int w_size = input[0][0][0].size();
    int flat_size = input[0].size() * input[0][0].size() * input[0][0][0].size();

    FCTensor output(batch_size, std::vector<float>(flat_size));

    for (int b = 0; b < batch_size; ++b)
    {
        int flat_index = 0;
        for (int c = 0; c < c_size; ++c)
        {
            for (int h = 0; h < h_size; ++h)
            {
                for (int w = 0; w < w_size; ++w)
                {
                    output[b][flat_index] = input[b][c][h][w];
                    flat_index += 1;
                }
            }
        }
    }
    return output;
}
Tensor AdaptiveAvgPool(const Tensor &input, int output_height, int output_width)
{
    int batch_size = input.size();
    int channels = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    float stride_y = static_cast<float>(input_height) / output_height;
    float stride_x = static_cast<float>(input_width) / output_width;

    Tensor output(batch_size, std::vector<std::vector<std::vector<float>>>(
                                  channels, std::vector<std::vector<float>>(
                                                output_height, std::vector<float>(
                                                                   output_width, 0.0))));

    for (int b = 0; b < batch_size; ++b)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < output_height; ++h)
            {
                for (int w = 0; w < output_width; ++w)
                {
                    int start_y = std::floor(h * stride_y);
                    int end_y = std::floor((h + 1) * stride_y);
                    int start_x = std::floor(w * stride_x);
                    int end_x = std::floor((w + 1) * stride_x);
                    end_y = std::min(end_y, input_height);
                    end_x = std::min(end_x, input_width);

                    float sum = 0.0;
                    for (int y = start_y; y < end_y; ++y)
                    {
                        for (int x = start_x; x < end_x; ++x)
                        {
                            sum += input[b][c][y][x];
                        }
                    }
                    int pool_area = (end_y - start_y) * (end_x - start_x);
                    output[b][c][h][w] = sum / pool_area;
                }
            }
        }
    }

    return output;
}
Tensor MaxPooling(const Tensor &input, int pool_size_x, int pool_size_y, int stride)
{
    int batch_size = input.size();
    int depth = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    int output_height = (input_height - pool_size_y) / stride + 1;
    int output_width = (input_width - pool_size_x) / stride + 1;

    Tensor output(batch_size, std::vector<std::vector<std::vector<float>>>(
                                  depth, std::vector<std::vector<float>>(
                                             output_height, std::vector<float>(
                                                                output_width, 0.0))));

    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < depth; ++d)
        {
            for (int h = 0; h < output_height; ++h)
            {
                for (int w = 0; w < output_width; ++w)
                {
                    float max_val = -FLT_MAX;
                    for (int i = 0; i < pool_size_y; ++i)
                    {
                        for (int j = 0; j < pool_size_x; ++j)
                        {
                            int h_index = h * stride + i;
                            int w_index = w * stride + j;
                            if (h_index < input_height && w_index < input_width)
                            {
                                max_val = std::max(max_val, input[b][d][h_index][w_index]);
                            }
                        }
                    }
                    output[b][d][h][w] = max_val;
                }
            }
        }
    }

    return output;
}
Tensor ReLU_conv(const Tensor &input)
{
    Tensor output = input;
    for (auto &batch : output)
    {
        for (auto &map : batch)
        {
            for (auto &row : map)
            {
                for (float &val : row)
                {
                    val = std::max(0.0f, val);
                }
            }
        }
    }
    return output;
}
FCTensor ReLU_FC(const FCTensor &input)
{
    FCTensor output = input;
    for (auto &batch : output)
    {
        for (auto &val : batch)
        {
            val = std::max(0.0f, val);
        }
    }
    return output;
}
FCTensor FullyConnected(const FCTensor &input, const WeightFC &weights, const WeightFCBias &bias)
{
    int batch_size = input.size();
    int output_features = weights.size();

    FCTensor output(batch_size, std::vector<float>(output_features, 0.0f));

    for (int b = 0; b < batch_size; ++b)
    {
        int mult_add_count = 0;
        for (int o = 0; o < output_features; ++o)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < input[b].size(); ++i)
            {
                sum += input[b][i] * weights[o][i];
                mult_add_count += 1;
            }
            output[b][o] = sum + bias[o];
            mult_add_count += 1;
        }
        cout << "mult_add_" << b << " : " << mult_add_count << endl;
    }
    return output;
}
Tensor ApplyPadding(Tensor &input, int padding)
{
    if (padding == 0)
        return input;
    int batch_size = input.size();
    int channels = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();

    Tensor output(batch_size, Pic(channels, std::vector<std::vector<float>>(height + 2 * padding, std::vector<float>(width + 2 * padding, 0.0))));

    for (int b = 0; b < batch_size; ++b)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    output[b][c][h + padding][w + padding] = input[b][c][h][w];
                }
            }
        }
    }

    return output;
}

Tensor Conv(WeightConv &weight_conv, WeightConvBias &weight_bias, Tensor &input, int kernel_x, int kernel_y, int ch_in, int ch_out, int stride, int padding)
{
    Tensor padded_input = ApplyPadding(input, padding);

    int batch_size = input.size();
    int input_height = padded_input[0][0].size();
    int input_width = padded_input[0][0][0].size();

    int output_height = (input_height - kernel_y) / stride + 1;
    int output_width = (input_width - kernel_x) / stride + 1;

    Tensor output(batch_size, Pic(ch_out, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0.0))));

    for (int b = 0; b < batch_size; ++b)
    {
        int mult_add_count = 0;
        for (int co = 0; co < ch_out; ++co)
        {
            for (int h = 0; h < output_height; ++h)
            {
                for (int w = 0; w < output_width; ++w)
                {
                    float sum = 0.0;
                    for (int ci = 0; ci < ch_in; ++ci)
                    {
                        for (int kh = 0; kh < kernel_y; ++kh)
                        {
                            for (int kw = 0; kw < kernel_x; ++kw)
                            {
                                if (h * stride + kh < input_height && w * stride + kw < input_width)
                                {
                                    sum += padded_input[b][ci][h * stride + kh][w * stride + kw] * weight_conv[co][ci][kh][kw];
                                    mult_add_count += 1;
                                }
                            }
                        }
                    }
                    sum += weight_bias[co];
                    output[b][co][h][w] = sum;
                    mult_add_count += 1;
                }
            }
        }
        cout << "mult_add_" << b << " : " << mult_add_count << endl;
    }
    return output;
}
std::vector<std::string> ReadLabel(const std::string &file_name)
{
    std::vector<std::string> label;
    std::ifstream file(file_name);
    std::string line;

    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file");
    }

    while (std::getline(file, line))
    {
        if (!line.empty())
        {
            label.push_back(line);
        }
    }

    if (label.empty())
    {
        std::cerr << "No labels were read from the file." << std::endl;
    }

    return label;
}
Pic ReadPic(const std::string &file_name, int c, int h, int w)
{
    Pic pic(c, std::vector<std::vector<float>>(h, std::vector<float>(w)));
    std::ifstream file(file_name);
    std::string line;
    float value;
    int channel = 0, row = 0, col = 0;

    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file");
    }

    while (file >> value)
    {
        pic[channel][row][col] = value;
        col++;
        if (col == w)
        {
            col = 0;
            row++;
            if (row == h)
            {
                row = 0;
                channel++;
                if (channel == c)
                {
                    break;
                }
            }
        }
    }
    file.close();
    return pic;
}
WeightFCBias ReadFCBiasWeight(const std::string file_path, int ch_out)
{
    WeightFCBias weight(ch_out, 0.0);
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return weight;
    }
    for (int out_channel = 0; out_channel < ch_out; ++out_channel)
    {
        if (!(file >> weight[out_channel]))
        {
            std::cerr << "Error reading weight for out_channel=" << out_channel << std::endl;
            return weight;
        }
    }
    file.close();
    return weight;
}
WeightFC ReadFCWeight(const std::string file_path, int ch_in, int ch_out)
{
    WeightFC weight(ch_out, std::vector<float>(
                                ch_in, 0.0));

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return weight;
    }

    for (int out_channel = 0; out_channel < ch_out; ++out_channel)
    {
        for (int in_channel = 0; in_channel < ch_in; ++in_channel)
        {
            if (!(file >> weight[out_channel][in_channel]))
            {
                std::cerr << "Error reading weight for out_channel=" << out_channel
                          << ", in_channel=" << in_channel << std::endl;
                return weight;
            }
        }
    }
    file.close();
    return weight;
}
WeightConvBias ReadConvBiasWeight(const std::string file_path, int ch_out)
{
    WeightConvBias weight(ch_out, 0.0);
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return weight;
    }
    for (int out_channel = 0; out_channel < ch_out; ++out_channel)
    {
        if (!(file >> weight[out_channel]))
        {
            std::cerr << "Error reading weight for out_channel=" << out_channel << std::endl;
            return weight;
        }
    }
    file.close();
    return weight;
}
WeightConv ReadConvWeight(const std::string file_path, int kernel_x, int kernel_y, int ch_in, int ch_out)
{
    WeightConv weight(ch_out, std::vector<std::vector<std::vector<float>>>(
                                  ch_in, std::vector<std::vector<float>>(
                                             kernel_y, std::vector<float>(kernel_x, 0.0))));

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return weight;
    }

    for (int out_channel = 0; out_channel < ch_out; ++out_channel)
    {
        for (int in_channel = 0; in_channel < ch_in; ++in_channel)
        {
            for (int y = 0; y < kernel_y; ++y)
            {
                for (int x = 0; x < kernel_x; ++x)
                {
                    if (!(file >> weight[out_channel][in_channel][y][x]))
                    {
                        std::cerr << "Error reading weight for out_channel=" << out_channel
                                  << ", in_channel=" << in_channel << ", y=" << y << ", x=" << x << std::endl;
                        return weight;
                    }
                }
            }
        }
    }

    file.close();
    return weight;
}
int sc_main(int argc, char *argv[])
{
    WeightConv conv1_weight, conv2_weight, conv3_weight, conv4_weight, conv5_weight;
    WeightConvBias conv1_bias, conv2_bias, conv3_bias, conv4_bias, conv5_bias;
    WeightFC fc6_weight, fc7_weight, fc8_weight;
    WeightFCBias fc6_bias, fc7_bias, fc8_bias;
    cout << "start loading weight\n";
    conv1_weight = ReadConvWeight("./conv1_weight.txt", 11, 11, 3, 64);
    conv2_weight = ReadConvWeight("./conv2_weight.txt", 5, 5, 64, 192);
    conv3_weight = ReadConvWeight("./conv3_weight.txt", 3, 3, 192, 384);
    conv4_weight = ReadConvWeight("./conv4_weight.txt", 3, 3, 384, 256);
    conv5_weight = ReadConvWeight("./conv5_weight.txt", 3, 3, 256, 256);
    conv1_bias = ReadConvBiasWeight("./conv1_bias.txt", 64);
    conv2_bias = ReadConvBiasWeight("./conv2_bias.txt", 192);
    conv3_bias = ReadConvBiasWeight("./conv3_bias.txt", 384);
    conv4_bias = ReadConvBiasWeight("./conv4_bias.txt", 256);
    conv5_bias = ReadConvBiasWeight("./conv5_bias.txt", 256);
    fc6_weight = ReadFCWeight("./fc6_weight.txt", 9216, 4096);
    fc7_weight = ReadFCWeight("./fc7_weight.txt", 4096, 4096);
    fc8_weight = ReadFCWeight("./fc8_weight.txt", 4096, 1000);
    fc6_bias = ReadFCBiasWeight("./fc6_bias.txt", 4096);
    fc7_bias = ReadFCBiasWeight("./fc7_bias.txt", 4096);
    fc8_bias = ReadFCBiasWeight("./fc8_bias.txt", 1000);
    cout << "weight loading finished\n";
    vector<string> label = ReadLabel("./imagenet_classes.txt");
    Tensor input_pic(2, std::vector<std::vector<std::vector<float>>>(
                            3, std::vector<std::vector<float>>(
                                   224, std::vector<float>(
                                            224, 0.0))));
    input_pic[0] = ReadPic("./dog.txt", 3, 224, 224);
    input_pic[1] = ReadPic("./cat.txt", 3, 224, 224);
    Tensor output;
    FCTensor FCoutput;
    printTensorSize(input_pic, "init");
    output = Conv(conv1_weight, conv1_bias, input_pic, 11, 11, 3, 64, 4, 2);
    printTensorSize(output, "2-1");
    output = ReLU_conv(output);
    printTensorSize(output, "2-2");
    output = MaxPooling(output, 3, 3, 2);
    printTensorSize(output, "2-3");
    output = Conv(conv2_weight, conv2_bias, output, 5, 5, 64, 192, 1, 2);
    printTensorSize(output, "2-4");
    output = ReLU_conv(output);
    printTensorSize(output, "2-5");
    output = MaxPooling(output, 3, 3, 2);
    printTensorSize(output, "2-6");
    output = Conv(conv3_weight, conv3_bias, output, 3, 3, 192, 384, 1, 1);
    printTensorSize(output, "2-7");
    output = ReLU_conv(output);
    printTensorSize(output, "2-8");
    output = Conv(conv4_weight, conv4_bias, output, 3, 3, 384, 256, 1, 1);
    printTensorSize(output, "2-9");
    output = ReLU_conv(output);
    printTensorSize(output, "2-10");
    output = Conv(conv5_weight, conv5_bias, output, 3, 3, 256, 256, 1, 1);
    printTensorSize(output, "2-11");
    output = ReLU_conv(output);
    printTensorSize(output, "2-12");
    output = MaxPooling(output, 3, 3, 2);
    printTensorSize(output, "2-13");
    output = AdaptiveAvgPool(output, 6, 6);
    printTensorSize(output, "1-2");
    FCoutput = Flatten(output);
    printFCTensorSize(FCoutput, "1-3");
    FCoutput = FullyConnected(FCoutput, fc6_weight, fc6_bias);
    printFCTensorSize(FCoutput, "2-15");
    FCoutput = ReLU_FC(FCoutput);
    printFCTensorSize(FCoutput, "2-16");
    FCoutput = FullyConnected(FCoutput, fc7_weight, fc7_bias);
    printFCTensorSize(FCoutput, "2-18");
    FCoutput = ReLU_FC(FCoutput);
    printFCTensorSize(FCoutput, "2-19");
    FCoutput = FullyConnected(FCoutput, fc8_weight, fc8_bias);
    printFCTensorSize(FCoutput, "2-20");
    printResult(FCoutput, label);
    // sc_signal<bool> clk, rst;
    // sc_signal<sc_uint<8>> pic;
    // sc_signal<sc_uint<10>> out;

    // ClockReset m_ClockReset("m_ClockReset");
    // AlexNet m_AlexNet("m_AlexNet");
    // Monitor m_Monitor("m_Monitor");
    // Pattern m_Pattern("m_Pattern");

    // m_ClockReset(clk, rst);
    // m_AlexNet(clk, rst, pic, out);
    // m_Monitor(clk, rst, pic, out);
    // m_Pattern(clk, rst, pic);

    // sc_start(500, SC_NS);
    // return 0;
}
