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