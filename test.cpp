#include <string>
#include "/usr/local/include/opencv2/objdetect.hpp"
#include "/usr/local/include/opencv2/highgui.hpp"
#include "/usr/local/include/opencv2/imgproc.hpp"

using namespace std;
using namespace cv;
int main( int argc, const char** argv) 
{
	string protoFile = "openpose/models/pose/COCO/pose_deploy_linevec.prototxt";
	string weightsFile = "openpose/models/pose/COCO/pose_iter_440000.caffemodel";
	
	Net net = readNetFromCaffe(protoFile,weightsFile);
}
