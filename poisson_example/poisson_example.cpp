// HalideTutorials.cpp : Defines the entry point for the console application.
//

#include <Halide.h>
#include "../halide_utils/halide_utils.h"

using namespace std;
using namespace Halide;
using namespace halide_utils;

int main()
{
	// TODO: Take image files as command-line arguments
	examples::poissonExample();
}
