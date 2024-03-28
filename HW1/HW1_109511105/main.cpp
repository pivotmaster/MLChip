#include "AlexNet.h"

using namespace std;
vector<float> softmax(const vector<float> &logits)
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

void printResult(FCTensor result, vector<string> label, vector<string> pic_name)
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
vector<string> ReadLabel(const string &file_name)
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
int sc_main(int argc, char *argv[])
{
    AlexNet alexnet("alexnet");
    sc_vector<sc_vector<sc_vector<sc_vector<sc_signal<float>>>>> input_image;
    sc_vector<sc_vector<sc_signal<float>>> output_distribution;
    const int PIC_NUM = argc - 1;
    alexnet.PIC_NUM = PIC_NUM;
    input_image.init(PIC_NUM);
    for (int b = 0; b < PIC_NUM; b++)
    {
        input_image[b].init(3);
        for (int i = 0; i < 3; i++)
        {
            input_image[b][i].init(224);
            for (int j = 0; j < 224; j++)
            {
                input_image[b][i][j].init(224);
            }
        }
    }
    for (int b = 0; b < PIC_NUM; b++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 224; j++)
                for (int k = 0; k < 224; k++)
                {
                    input_image[b][i][j][k].write(123.0);
                }
    vector<string> pic_name(PIC_NUM, "");
    pic_name[0] = argv[1];
    if (PIC_NUM == 2)
        pic_name[1] = argv[2];
    for (int b = 0; b < PIC_NUM; b++)
    {
        ifstream file(pic_name[b]);
        string line;
        float value;
        int channel = 0, row = 0, col = 0;

        if (!file.is_open())
        {
            throw runtime_error("Unable to open file");
        }

        while (file >> value)
        {
            input_image[b][channel][row][col] = value;
            col++;
            if (col == 224)
            {
                col = 0;
                row++;
                if (row == 224)
                {
                    row = 0;
                    channel++;
                    if (channel == 3)
                    {
                        break;
                    }
                }
            }
        }
        file.close();
    }
    for (int b = 0; b < PIC_NUM; b++)
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 224; j++)
            {
                for (int k = 0; k < 224; k++)
                {
                    alexnet.input_data[b][i][j][k](input_image[b][i][j][k]);
                }
            }
        }
    output_distribution.init(PIC_NUM);
    for (int i = 0; i < PIC_NUM; i++)
    {
        output_distribution[i].init(1000);
    }
    for (int i = 0; i < PIC_NUM; i++)
    {
        for (int j = 0; j < 1000; j++)
            alexnet.output_data[i][j](output_distribution[i][j]);
    }
    sc_start();
    vector<string> label = ReadLabel("./imagenet_classes.txt");
    FCTensor result(PIC_NUM, vector<float>(1000, 0.0));
    for (int b = 0; b < PIC_NUM; b++)
    {
        for (int c = 0; c < 1000; c++)
            result[b][c] = output_distribution[b][c];
    }
    printResult(result, label, pic_name);
    return 0;
}
