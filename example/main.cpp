#include "MouseExample.h"
#include "examples.h"

#ifdef BUILD_CARS_COUNTING
#include "CarsCounting.h"
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

// ----------------------------------------------------------------------

static void Help()
{
    printf("\nExamples of the Multitarget tracking algorithm\n"
           "Usage: \n"
           "          ./MultitargetTracker <path to movie file> [--example]=<number of example 0..7> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> [--async]=<async pipeline> [--res]=<csv log file> [--settings]=<ini file> [--batch_size=<number of frames>] \n\n"
           "Press:\n"
           "\'m\' key for change mode: play|pause. When video is paused you can press any key for get next frame. \n\n"
           "Press Esc to exit from video \n\n"
           );
}

const char* keys =
{
    "{ @1               |C:/multitarget-tracker-modify/data/highway4.mp4  | movie file | }"
    "{ e  example       |7                   | number of example 0 - MouseTracking, 1 - MotionDetector, 2 - FaceDetector, 3 - PedestrianDetector, 4 - OpenCV dnn objects detector, 5 - YOLO Darknet detector, 6 - YOLO TensorRT Detector, 7 - Cars counting | }"
    "{ sf start_frame   |0                   | Start a video from this position | }"
    "{ ef end_frame     |0                   | Play a video to this position (if 0 then played to the end of file) | }"
    "{ ed end_delay     |0                   | Delay in milliseconds after video ending | }"
    "{ o  out           |                    | Name of result video file | }"
    "{ sl show_logs     |1                   | Show Trackers logs | }"
    "{ g gpu            |0                   | Use OpenCL acceleration | }"
    "{ a async          |1                   | Use 2 theads for processing pipeline | }"
    "{ r res            |                    | Path to the csv file with tracking result | }"
    "{ s settings       |                    | Path to the init file with tracking settings | }"
    "{ bs batch_size    |1                   | Batch size - frames count for processing | }"
    "{ inf inference    |darknet             | For CarsCounting: Type of inference framework: darknet, ocvdnn | }"
    "{ w weights        |                    | For CarsCounting: Weights of neural network: yolov4.weights | }"
    "{ c config         |                    | For CarsCounting: Config file of neural network: yolov4.cfg | }"
    "{ n names          |                    | For CarsCounting: File with classes names: coco.names | }"
    "{ wf write_n_frame |1                   | Write logs on each N frame: 1 for writing each frame | }"
    "{ hm heat_map      |0                   | For CarsCounting: Draw heat map | }"
    "{ gb geo_bind      |C:/multitarget-tracker-modify/data/map/geo_bind_lkh.ini        | For CarsCounting: ini file with geographical binding | }"
    "{ rt rtsp          |0                   | Source from IP camera through RTSP }"
    "{ nx onnx          |0                   | Network model type: 0 - not ONNX, 6 - yolov6, 7 - yolov7 | }"
};

// ----------------------------------------------------------------------

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    Help();
    parser.printMessage();

    bool useOCL = parser.get<int>("gpu") != 0;
    cv::ocl::setUseOpenCL(useOCL);
    std::cout << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

    int exampleNum = parser.get<int>("example");
    int asyncPipeline = parser.get<int>("async");

	std::unique_ptr<VideoExample> detector;

    switch (exampleNum)
    {
    case 0:
        MouseTracking(parser);
        break;

    case 1:
        detector = std::make_unique<MotionDetectorExample>(parser);
        break;

    case 2:
		detector = std::make_unique<FaceDetectorExample>(parser);
        break;

    case 3:
		detector = std::make_unique<PedestrianDetectorExample>(parser);
        break;

	case 4:
		detector = std::make_unique<OpenCVDNNExample>(parser);
		break;

#ifdef BUILD_YOLO_LIB
	case 5:
		detector = std::make_unique<YoloDarknetExample>(parser);
		break;
#endif

#ifdef BUILD_YOLO_TENSORRT
	case 6:
		detector = std::make_unique<YoloTensorRTExample>(parser);
		break;
#endif

#ifdef BUILD_CARS_COUNTING
    case 7:
    {
        auto carsCounting = new CarsCounting(parser);
        detector = std::unique_ptr<CarsCounting>(carsCounting);
        break;
    }
#endif

    default:
        std::cerr << "Wrong example number: " << exampleNum << std::endl;
        break;
    }

    if (detector.get())
    {
#ifdef FULL_PIPELINE_PROCESS
        asyncPipeline ? detector->PipelineProcess() : detector->SyncProcess();
#else
        asyncPipeline ? detector->AsyncProcess() : detector->SyncProcess();
#endif
    }

#ifndef SILENT_WORK
    cv::destroyAllWindows();
#endif
    return 0;
}
