#include "AlexNet.h"

using namespace std;

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
