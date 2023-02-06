// #include "yolov5.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include "NvInfer.h"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>


#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define CONF_THRESH 0.25
#define BATCH_SIZE 1

static constexpr int MAX_OUTPUT_BBOX_COUNT = 25200;
static constexpr int CLASS_NUM = 1;
static constexpr int INPUT_H = 640;
static constexpr int INPUT_W = 640;
static constexpr int IMG_H = 640;
static constexpr int IMG_W = 640;
static const int OUTPUT_SIZE = MAX_OUTPUT_BBOX_COUNT * 6 + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_DET_NAME = "det";
const char* OUTPUT_SEG_NAME = "seg";
const char* OUTPUT_LANE_NAME = "lane";

class Logger : public nvinfer1::ILogger
{
public:
  explicit Logger(bool verbose) : verbose_(verbose) {}

  void log(Severity severity, const char * msg) noexcept override
  {
    if (verbose_ || ((severity != Severity::kINFO) && (severity != Severity::kVERBOSE))) {
      std::cout << msg << std::endl;
    }
  }

private:
  bool verbose_{false};
};

static Logger gLogger(false);

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    cv::Mat tensor;
    out.convertTo(tensor, CV_32FC3, 1.f / 255.f);

    cv::subtract(tensor, cv::Scalar(0.485, 0.456, 0.406), tensor, cv::noArray(), -1);
    cv::divide(tensor, cv::Scalar(0.229, 0.224, 0.225), tensor, 1, -1);
    // std::cout << cv::format(out, cv::Formatter::FMT_NUMPY)<< std::endl;
    // assert(false);
    // cv::Mat out(input_h, input_w, CV_8UC3);
    // cv::copyMakeBorder(re, out, y, y, x, x, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    return tensor;
}

void doInferenceCpu(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* det_output, int* seg_output, int* lane_output, int batchSize) 
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(det_output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(seg_output, buffers[2], batchSize * IMG_H * IMG_W * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(lane_output, buffers[3], batchSize * IMG_H * IMG_W * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

// void doInference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void **buffers, float* det_output, int* seg_output, int* lane_output, int batchSize) {
//     // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//     // CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//     context.enqueue(batchSize, buffers, stream, nullptr);
//     CUDA_CHECK(cudaMemcpyAsync(det_output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaMemcpyAsync(seg_output, buffers[2], batchSize * IMG_H * IMG_W * sizeof(int), cudaMemcpyDeviceToHost, stream));
//     CUDA_CHECK(cudaMemcpyAsync(lane_output, buffers[3], batchSize * IMG_H * IMG_W * sizeof(int), cudaMemcpyDeviceToHost, stream));
//     cudaStreamSynchronize(stream);
// }

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;

    void print()
    {
        std::cout<<"conf:"<<conf<<std::endl;
        std::cout<<"c_x:"<<bbox[0]
            <<",c_y:"<<bbox[1]
            <<",w:"<<bbox[2]
            <<",h:"<<bbox[3]<<std::endl;
    }
};
bool cmp(const Detection& a, const Detection& b) {
    return a.conf > b.conf;
}
float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

int main(int argc, char** argv)
{
    // deserialize the .engine and run inference
    std::string engine_name = "../yolop.engine";
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
       data[i] = 1.0;
    static float det_out[BATCH_SIZE * OUTPUT_SIZE];
    static int seg_out[BATCH_SIZE * IMG_H * IMG_W];
    static int lane_out[BATCH_SIZE * IMG_H * IMG_W];
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 4);
    void* buffers[4];
    // // In order to bind the buffers, we need to know the names of the input and output tensors.
    // // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    // const int output_det_index = engine->getBindingIndex(OUTPUT_DET_NAME);
    // const int output_seg_index = engine->getBindingIndex(OUTPUT_SEG_NAME);
    // const int output_lane_index = engine->getBindingIndex(OUTPUT_LANE_NAME);
    // assert(inputIndex == 0);
    // assert(output_det_index == 1);
    // assert(output_seg_index == 2);
    // assert(output_lane_index == 3);

    const int inputIndex = 0;
    const int output_det_index = 1;
    const int output_seg_index = 2;
    const int output_lane_index = 3;

    // // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_det_index], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_seg_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers[output_lane_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    // // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // store seg results
    cv::Mat tmp_seg(IMG_H, IMG_W, CV_32S, seg_out);
    // store lane results
    cv::Mat tmp_lane(IMG_H, IMG_W, CV_32S, lane_out);
    // PrintMat(tmp_seg);
    std::vector<cv::Vec3b> segColor;
    segColor.push_back(cv::Vec3b(0, 0, 0));
    segColor.push_back(cv::Vec3b(0, 255, 0));
    segColor.push_back(cv::Vec3b(255, 0, 0));

    std::vector<cv::Vec3b> laneColor;
    laneColor.push_back(cv::Vec3b(0, 0, 0));
    laneColor.push_back(cv::Vec3b(0, 0, 255));
    laneColor.push_back(cv::Vec3b(0, 0, 0));

    // preprocess ~3ms
    std::string imgpath = "/home/yolop_tensorrt/test.jpg";
    cv::Mat srcimg = cv::imread(imgpath);
    cv::Mat pr_img = preprocess_img(srcimg, INPUT_W, INPUT_H); // letterbox

    // BGR to RGB and normalize
    int i= 0;
    for (int row = 0; row < INPUT_H; ++row) {
        float* uc_pixel = pr_img.ptr<float>(row);
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = uc_pixel[0];
            data[i + INPUT_H * INPUT_W] = uc_pixel[1];
            data[i + 2 * INPUT_H * INPUT_W] = uc_pixel[2];
            uc_pixel += 3;
            ++i;
        }
    }
    
    // Run inference
    auto start = std::chrono::system_clock::now();
    doInferenceCpu(*context, stream, buffers, data, det_out, seg_out, lane_out, BATCH_SIZE);
    
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<Detection> res;
    nms(res, &det_out[0], CONF_THRESH, NMS_THRESH);
    for(auto d:res)
    {
        d.print();
    }
    cv::Mat img = cv::imread(imgpath);

    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    // handling seg and lane results    
    cv::Mat outimg = out.clone();
	int area = INPUT_H * INPUT_W;
	for (int i = 0; i < outimg.rows; i++)
	{
		for (int j = 0; j < outimg.cols; j++)
		{
			int x = i;
            int y = j;
            if (seg_out[y * INPUT_W + x] < seg_out[area + y * INPUT_W + x])
			{
				outimg.at<cv::Vec3b>(i, j)[0] = 0;
				outimg.at<cv::Vec3b>(i, j)[1] = 255;
				outimg.at<cv::Vec3b>(i, j)[2] = 0;
			}
		}
	}
    cv::imwrite("../out.jpg", outimg);

    cv::imwrite("../results.jpg", img);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[output_det_index]));
    CUDA_CHECK(cudaFree(buffers[output_seg_index]));
    CUDA_CHECK(cudaFree(buffers[output_lane_index]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}