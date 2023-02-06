#include "yolov5.hpp"
#include <csignal>

static volatile bool keep_running = true;


void keyboard_handler(int sig) {
    // handle keyboard interrupt
    if (sig == SIGINT)
        keep_running = false;
}


int main(int argc, char** argv) 
{
    signal(SIGINT, keyboard_handler);
    cudaSetDevice(DEVICE);

    std::string wts_name = "yolop.wts";
    std::string engine_name = "yolop.engine";

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        std::cout << "Building engine..." << std::endl;
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        std::cout << "Engine has been built and saved to file." << std::endl;
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

    return 0;
}
