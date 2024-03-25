#include "AlexNet.h"

using namespace std;

vector<float> AlexNet::softmax(const vector<float> &logits)
{
    vector<float> exp_values;
    transform(logits.begin(), logits.end(), back_inserter(exp_values), [](float logit)
              { return exp(logit); });

    float sum = accumulate(exp_values.begin(), exp_values.end(), 0.0f);
    vector<float> probabilities;
    transform(exp_values.begin(), exp_values.end(), back_inserter(probabilities), [sum](float exp_value)
              { return exp_value / sum; });

    return probabilities;
}
void AlexNet::printResult(FCTensor result, vector<string> label, vector<string> pic_name)
{
    for (int i = 0; i < result.size(); i++)
    {
        cout << endl
             << pic_name[i] << endl;
        cout << "====================================================" << endl;
        cout << " idx |      val |   possibility |  class name       " << endl;
        vector<float> possibility = softmax(result[i]);
        vector<tuple<int, float, float>> indexedPossibilityWithScore;
        for (int j = 0; j < possibility.size(); j++)
        {
            indexedPossibilityWithScore.push_back(make_tuple(j, possibility[j], result[i][j]));
        }
        vector<tuple<int, float, float>> topTen(10);
        partial_sort_copy(indexedPossibilityWithScore.begin(), indexedPossibilityWithScore.end(), topTen.begin(), topTen.end(),
                          [](const tuple<int, float, float> &left, const tuple<int, float, float> &right)
                          {
                              return get<1>(left) > get<1>(right);
                          });
        for (auto &[index, val, originalScore] : topTen)
        {
            cout << " " << setw(3) << index << " |  " << setw(5) << fixed << setprecision(4) << originalScore << " |    " << setw(10) << setprecision(5) << val * 100 << " | "
                 << label[index] << endl;
        }
    }
}
void AlexNet::printFCTensorSize(const FCTensor &input, const string &name)
{
    int batch_size = input.size();
    int depth = input[0].size();

    cout << left << setw(5) << name << " size: ";
    cout << right << setw(6) << batch_size << setw(6) << depth << "\n";
}
void AlexNet::printTensorSize(const Tensor &input, const string &name)
{
    int batch_size = input.size();
    int depth = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    cout << left << setw(5) << name << " size: ";
    cout << right << setw(6) << batch_size << setw(6) << depth << setw(6) << input_height << setw(6) << input_width << "\n";
}
FCTensor AlexNet::Flatten(const Tensor &input)
{
    int batch_size = input.size();
    int c_size = input[0].size();
    int h_size = input[0][0].size();
    int w_size = input[0][0][0].size();
    int flat_size = input[0].size() * input[0][0].size() * input[0][0][0].size();

    FCTensor output(batch_size, vector<float>(flat_size));

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
Tensor AlexNet::AdaptiveAvgPool(const Tensor &input, int output_height, int output_width)
{
    int batch_size = input.size();
    int channels = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    float stride_y = static_cast<float>(input_height) / output_height;
    float stride_x = static_cast<float>(input_width) / output_width;

    Tensor output(batch_size, vector<vector<vector<float>>>(
                                  channels, vector<vector<float>>(
                                                output_height, vector<float>(
                                                                   output_width, 0.0))));

    for (int b = 0; b < batch_size; ++b)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < output_height; ++h)
            {
                for (int w = 0; w < output_width; ++w)
                {
                    int start_y = floor(h * stride_y);
                    int end_y = floor((h + 1) * stride_y);
                    int start_x = floor(w * stride_x);
                    int end_x = floor((w + 1) * stride_x);
                    end_y = min(end_y, input_height);
                    end_x = min(end_x, input_width);

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
Tensor AlexNet::MaxPooling(const Tensor &input, int pool_size_x, int pool_size_y, int stride)
{
    int batch_size = input.size();
    int depth = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    int output_height = (input_height - pool_size_y) / stride + 1;
    int output_width = (input_width - pool_size_x) / stride + 1;

    Tensor output(batch_size, vector<vector<vector<float>>>(
                                  depth, vector<vector<float>>(
                                             output_height, vector<float>(
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
                                max_val = max(max_val, input[b][d][h_index][w_index]);
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
Tensor AlexNet::ReLU_conv(const Tensor &input)
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
                    val = max(0.0f, val);
                }
            }
        }
    }
    return output;
}
FCTensor AlexNet::ReLU_FC(const FCTensor &input)
{
    FCTensor output = input;
    for (auto &batch : output)
    {
        for (auto &val : batch)
        {
            val = max(0.0f, val);
        }
    }
    return output;
}
FCTensor AlexNet::FullyConnected(const FCTensor &input, const WeightFC &weights, const WeightFCBias &bias)
{
    int batch_size = input.size();
    int output_features = weights.size();

    FCTensor output(batch_size, vector<float>(output_features, 0.0f));

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
Tensor AlexNet::ApplyPadding(Tensor &input, int padding)
{
    if (padding == 0)
        return input;
    int batch_size = input.size();
    int channels = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();

    Tensor output(batch_size, Pic(channels, vector<vector<float>>(height + 2 * padding, vector<float>(width + 2 * padding, 0.0))));

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

Tensor AlexNet::Conv(WeightConv &weight_conv, WeightConvBias &weight_bias, Tensor &input, int kernel_x, int kernel_y, int ch_in, int ch_out, int stride, int padding)
{
    Tensor padded_input = ApplyPadding(input, padding);

    int batch_size = input.size();
    int input_height = padded_input[0][0].size();
    int input_width = padded_input[0][0][0].size();

    int output_height = (input_height - kernel_y) / stride + 1;
    int output_width = (input_width - kernel_x) / stride + 1;

    Tensor output(batch_size, Pic(ch_out, vector<vector<float>>(output_height, vector<float>(output_width, 0.0))));

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
vector<string> AlexNet::ReadLabel(const string &file_name)
{
    vector<string> label;
    ifstream file(file_name);
    string line;

    if (!file.is_open())
    {
        throw runtime_error("Unable to open file");
    }

    while (getline(file, line))
    {
        if (!line.empty())
        {
            label.push_back(line);
        }
    }

    if (label.empty())
    {
        cerr << "No labels were read from the file." << endl;
    }

    return label;
}
Pic AlexNet::ReadPic(const string &file_name, int c, int h, int w)
{
    Pic pic(c, vector<vector<float>>(h, vector<float>(w)));
    ifstream file(file_name);
    string line;
    float value;
    int channel = 0, row = 0, col = 0;

    if (!file.is_open())
    {
        throw runtime_error("Unable to open file");
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
WeightFCBias AlexNet::ReadFCBiasWeight(const string file_path, int ch_out)
{
    WeightFCBias weight(ch_out, 0.0);
    ifstream file(file_path);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << file_path << endl;
        return weight;
    }
    for (int out_channel = 0; out_channel < ch_out; ++out_channel)
    {
        if (!(file >> weight[out_channel]))
        {
            cerr << "Error reading weight for out_channel=" << out_channel << endl;
            return weight;
        }
    }
    file.close();
    return weight;
}
WeightFC AlexNet::ReadFCWeight(const string file_path, int ch_in, int ch_out)
{
    WeightFC weight(ch_out, vector<float>(
                                ch_in, 0.0));

    ifstream file(file_path);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << file_path << endl;
        return weight;
    }

    for (int out_channel = 0; out_channel < ch_out; ++out_channel)
    {
        for (int in_channel = 0; in_channel < ch_in; ++in_channel)
        {
            if (!(file >> weight[out_channel][in_channel]))
            {
                cerr << "Error reading weight for out_channel=" << out_channel
                     << ", in_channel=" << in_channel << endl;
                return weight;
            }
        }
    }
    file.close();
    return weight;
}
WeightConvBias AlexNet::ReadConvBiasWeight(const string file_path, int ch_out)
{
    WeightConvBias weight(ch_out, 0.0);
    ifstream file(file_path);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << file_path << endl;
        return weight;
    }
    for (int out_channel = 0; out_channel < ch_out; ++out_channel)
    {
        if (!(file >> weight[out_channel]))
        {
            cerr << "Error reading weight for out_channel=" << out_channel << endl;
            return weight;
        }
    }
    file.close();
    return weight;
}
WeightConv AlexNet::ReadConvWeight(const string file_path, int kernel_x, int kernel_y, int ch_in, int ch_out)
{
    WeightConv weight(ch_out, vector<vector<vector<float>>>(
                                  ch_in, vector<vector<float>>(
                                             kernel_y, vector<float>(kernel_x, 0.0))));

    ifstream file(file_path);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << file_path << endl;
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
                        cerr << "Error reading weight for out_channel=" << out_channel
                             << ", in_channel=" << in_channel << ", y=" << y << ", x=" << x << endl;
                        return weight;
                    }
                }
            }
        }
    }

    file.close();
    return weight;
}
FCTensor AlexNet::inference(Tensor &input_pic)
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
    return FCoutput;
}

vector<vector<vector<float>>> in_data;
vector<vector<float>> out_data;
const int PIC_NUM = 2;

void AlexNet::process()
{
    vector<string> pic_name(PIC_NUM, "");
    //////////////////////
    // change pic names //
    pic_name[0] = "dog.txt";
    pic_name[1] = "cat.txt";
    //////////////////////
    in_data.resize(3, vector<vector<float>>(224, vector<float>(224, 0.0)));
    out_data.resize(PIC_NUM, vector<float>(1000, 0.0));

    Tensor input_pic(PIC_NUM, vector<vector<vector<float>>>(
                                  3, vector<vector<float>>(
                                         224, vector<float>(
                                                  224, 0.0))));
    input_pic[0] = ReadPic(pic_name[0], 3, 224, 224);
    if (PIC_NUM == 2)
        input_pic[1] = ReadPic(pic_name[1], 3, 224, 224);
    vector<vector<float>> output = inference(input_pic);
    for (int p = 0; p < PIC_NUM; p++)
    {
        for (int i = 0; i < 1000; i++)
        {
            out_data[p][i] = output[p][i];
        }
    }
    vector<string> label = ReadLabel("./imagenet_classes.txt");
    printResult(out_data, label, pic_name);
    exit(0);
}
