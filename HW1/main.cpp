#include "AlexNet.h"
// #include "Pattern.h"
#include "Monitor.h"
#include "clockreset.h"

using namespace std;

int sc_main(int argc, char *argv[])
{
    sc_clock clk("clk", 1, SC_NS);
    AlexNet alexnet("alexnet");
    sc_start();
    return 0;
}
