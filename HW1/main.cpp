#include "AlexNet.h"

using namespace std;

int sc_main(int argc, char *argv[])
{
    sc_clock clk("clk", 1, SC_NS);
    AlexNet alexnet("alexnet");
    sc_start();
    return 0;
}
