#include "AlexNet.h"

using namespace std;

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