#include "AlexNet.h"

using namespace std;

void AlexNet::printTensorSize(const Tensor &input, const string &name)
{
	int batch_size = input.size();
	int depth = input[0].size();
	int input_height = input[0][0].size();
	int input_width = input[0][0][0].size();

	cout << left << setw(5) << name << " size: ";
	cout << right << setw(6) << batch_size << setw(6) << depth << setw(6) << input_height << setw(6) << input_width << "\n";
}

void AlexNet::printFCTensorSize(const FCTensor &input, const string &name)
{
	int batch_size = input.size();
	int depth = input[0].size();

	cout << left << setw(5) << name << " size: ";
	cout << right << setw(6) << batch_size << setw(6) << depth << "\n";
}
