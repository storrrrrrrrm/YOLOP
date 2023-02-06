#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include "NvInfer.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 1;
    static constexpr int INPUT_H = 384;
    static constexpr int INPUT_W = 640;
    static constexpr int IMG_H = 360;
    static constexpr int IMG_W = 640;
    // static constexpr int INPUT_H = 192;
    // static constexpr int INPUT_W = 320;
    // static constexpr int IMG_H = 180;
    // static constexpr int IMG_W = 320;

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2IOExt
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int getNbOutputs() 
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept;

        int initialize() noexcept;

        virtual void terminate() noexcept {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept { return 0; }

        // virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() noexcept;

        virtual void serialize(void* buffer) noexcept;

        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) 
        {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType();

        const char* getPluginVersion();

        void destroy() noexcept;

        IPluginV2IOExt* clone() noexcept;

        void setPluginNamespace(const char* pluginNamespace) noexcept;

        const char* getPluginNamespace() noexcept;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) noexcept;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) noexcept;

        bool canBroadcastInputAcrossBatch(int inputIndex) noexcept;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept;

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept;

        void detachFromContext() noexcept;

    private:
        void forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() noexcept;

        const char* getPluginVersion() const noexcept;

        const PluginFieldCollection* getFieldNames() noexcept;

        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept;

        void setPluginNamespace(const char* libNamespace) noexcept
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const noexcept
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 
