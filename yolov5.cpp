/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "sampleEngines.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_fp16.h>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.sample_onnx_mnist";
const std::vector<std::string> classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
const std::vector<cv::Scalar> colors = {cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255), cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61), cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0), cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132), cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)};
//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    cv::Mat img_;
    float ratio_ = 0.0;

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool postProcess(const samplesCommon::BufferManager& buffers);

    bool NMS(std::vector<std::vector<float>>& output, float conf_thres, float iou_thres, int max_det, std::vector<std::vector<float>>& nms_result);
    
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    if (mParams.loadEngine.empty()) {
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder)
        {
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
        {
            return false;
        }

        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            return false;
        }

        auto constructed = constructNetwork(builder, network, config);
        if (!constructed)
        {
            return false;
        }

        // CUDA stream used for profiling by the builder.
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream)
        {
            return false;
        }
        config->setProfileStream(*profileStream);

        SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            return false;
        }

        mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
        if (!mRuntime)
        {
            return false;
        }

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    } else {
        // mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        //     sample::loadEngine(mParams.loadEngine, mParams.dlaCore, std::cerr), samplesCommon::InferDeleter());
        std::ifstream planFile(mParams.loadEngine);

        if (!planFile.is_open())
        {
            sample::gLogError << "Could not open plan file: " << mParams.loadEngine << std::endl;
            return false;
        }

        std::stringstream planBuffer;
        planBuffer << planFile.rdbuf();
        std::string plan = planBuffer.str();
        const auto& mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
        if (!mRuntime)
        {
            return false;
        }
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(plan.data(), plan.size()), samplesCommon::InferDeleter());
    }

    if (!mEngine)
    {
        sample::gLogError << "load engine failed!" << std::endl;
        return false;
    }
    // IHostMemory* serialized_model (mEngine->serialize());
    // std::ofstream f("yolov5s.engine", std::ios::binary);
    // f.write(reinterpret_cast<char*>(serialized_model->data()), serialized_model->size());
    // serialized_model->destroy();
    if (!mParams.saveEngine.empty())
    {
        sample::gLogInfo << "Saving engine to: " << mParams.saveEngine << std::endl;
        sample::saveEngine(*mEngine, mParams.saveEngine, std::cerr);
    }

    // ASSERT(network->getNbInputs() == 1);
    // mInputDims = network->getInput(0)->getDimensions();
    // ASSERT(mInputDims.nbDims == 4);

    // ASSERT(network->getNbOutputs() == 1);
    // mOutputDims = network->getOutput(0)->getDimensions();
    sample::gLogInfo << "bindings num: " << mEngine->getNbBindings() << std::endl;
    for (int i = 0; i < mEngine->getNbBindings(); ++i)
    {
        nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
        bool is_input = mEngine->bindingIsInput(i);
        std::string binding_name = mEngine->getBindingName(i);

        std::cout << "Binding " << i << " (" << binding_name << "): ";
        if(is_input) {
            mInputDims = dims;
            std::cout << "input,  ";
        } else {
            mOutputDims = dims;
            std::cout << "onput,  ";
        }
        std::cout << "dimensions: ";
        for(int j=0; j<dims.nbDims; ++j)
        {
            std::cout << dims.d[j] << ' ';
        }
        std::cout << '\n';

        nvinfer1::DataType trtType = mEngine->getBindingDataType(i);

        switch(trtType)
        {
            case nvinfer1::DataType::kFLOAT:
                std::cout << "Binding " << i << ": FLOAT" << std::endl;
                break;
            case nvinfer1::DataType::kHALF:
                std::cout << "Binding " << i << ": HALF" << std::endl;
                break;
            case nvinfer1::DataType::kINT8:
                std::cout << "Binding " << i << ": INT8" << std::endl;
                break;
            case nvinfer1::DataType::kINT32:
                std::cout << "Binding " << i << ": INT32" << std::endl;
                break;
            default:
                std::cout << "Unknown type for binding " << i << std::endl;
                break;
        }
    }

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
    printf("fp16: %d, int8: %d\n", mParams.fp16, mParams.int8);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    auto start = std::chrono::high_resolution_clock::now();
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    sample::gLogInfo << "processInput success in: " << duration.count() << " ms." << std::endl;

    start = std::chrono::high_resolution_clock::now();
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    sample::gLogInfo << "engine inference in: " << duration.count() << " ms." << std::endl;

    if (!status)
    {
        return false;
    }

    start = std::chrono::high_resolution_clock::now();
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!postProcess(buffers))
    {
        return false;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    sample::gLogInfo << "postProcess success in: " << duration.count() << " ms." << std::endl;

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    const int inputC = mInputDims.d[1];
    cv::Mat img = cv::imread(mParams.source, cv::IMREAD_COLOR);
    if (img.empty()) {
        sample::gLogError << mParams.source << "read result is empty!" << std::endl;
        return false;
    }
    img_ = img;
    int height = img.rows;
    int width = img.cols;
    if (height == 0 || width == 0) {
        sample::gLogError << "weight or height equal 0..." << std::endl;
        return false;
    }
    // printf("inputH: %d, inputW: %d, inputC: %d, height: %d, width: %d\n", inputH, inputW, inputC, height, width);
    float h_ratio = (float)inputH / height;
    float w_ratio = (float)inputW / width;
    float ratio = std::min(h_ratio, w_ratio);
    ratio_ = ratio;
    // printf("ratio: %f\n", ratio);
    int new_unpad_h = std::round(height * ratio);
    int new_unpad_w = std::round(width * ratio);
    // printf("new_unpad_h: %d, new_unpad_w: %d\n", new_unpad_h, new_unpad_w);
    int dw = inputW - new_unpad_w;
    int dh = inputH - new_unpad_h;
    // printf("dw: %d, dh: %d\n", dw, dh);
    dw /= 2; dh /= 2;
    if (height != new_unpad_h || width != new_unpad_w) {
        cv::resize(img, img, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    }
    int top = std::round(dh - 0.1);
    int bottom = std::round(dh + 0.1);
    int left = std::round(dw - 0.1);
    int right = std::round(dw + 0.1);
    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (img.rows != inputH || img.cols != inputW || img.channels() != inputC) {
        printf("not equal input (%d, %d, %d), height: %d, width: %d, chanenl: %d", inputH, inputW, inputC, img.rows, img.cols, img.channels());
        return false;
    }
    // 1. BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 2. HWC to CHW
    cv::Mat imgCHW;
    cv::transpose(img.reshape(1, img.total()), imgCHW);
    // printf("=======================================================================");
    // printf("h: %d, w: %d, c: %d\n", imgCHW.rows, imgCHW.cols, imgCHW.channels());

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for(int i = 0; i < imgCHW.rows; ++i)
    {
        for(int j = 0; j < imgCHW.cols; ++j)
        {
            hostDataBuffer[i * imgCHW.cols + j] = static_cast<float>(imgCHW.at<uchar>(i,j)) / 255;
        }
    }
    // for (int i = 1000000; i < 1000100; i++) {
    //     std::cout << hostDataBuffer[i] << ", ";
    // }
    // std::cout << std::endl;

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::postProcess(const samplesCommon::BufferManager& buffers)
{
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float conf_thres = 0.25;
    float iou_thres = 0.45;
    int max_det = 1000;
    // for (int i = 0; i < 100; i++) {
    //     std::cout << output[i] << ", ";
    // }
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    const int outputH = mOutputDims.d[1];
    const int outputW = mOutputDims.d[2];
    std::vector<std::vector<float>> matrix(outputH, std::vector<float>(outputW, 0));
    for (int i = 0; i < outputH; i++) {
        for (int j = 0; j < outputW; j++) {
            matrix[i][j] = output[i * outputW + j];
        }
    }
    // printf("last value: %f\n", matrix[outputH - 1][outputW - 1]);
    std::vector<std::vector<float>> nms_result;
    bool nms_ret = NMS(matrix, conf_thres, iou_thres, max_det, nms_result);
    if (!nms_ret || nms_result.size() == 0 || nms_result[0].size() == 0) {
        sample::gLogError << "nms result empty!" << std::endl;
        return false;
    }
    // printf("nms_result: (%lu, %lu)\n", nms_result.size(), nms_result[0].size());
    // for (int i = 0; i < nms_result.size(); i++) {
    //     for (int j = 0; j < nms_result[0].size(); j++) {
    //         printf("%f, ", nms_result[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    float pad_w = (inputW - img_.cols * ratio_) / 2;
    float pad_h = (inputH - img_.rows * ratio_) / 2;
    // printf("pad_w: %f, pad_h : %f\n", pad_w, pad_h);
    for (int i = 0; i < nms_result.size(); i++) {
        nms_result[i][0] = (nms_result[i][0] - pad_w) / ratio_;
        nms_result[i][2] = (nms_result[i][2] - pad_w) / ratio_;
        nms_result[i][1] = (nms_result[i][1] - pad_h) / ratio_;
        nms_result[i][3] = (nms_result[i][3] - pad_h) / ratio_;
    }

    std::vector<std::vector<int>> boxes(nms_result.size(), std::vector<int>(4, 0));

    int lw = 3;
    int tf = std::max(lw - 1, 1);
    int fontFace = cv::FONT_HERSHEY_SIMPLEX; // 字体类型
    double fontScale = lw / 3;
    int baseLine = 0;

    for (int i = 0; i < boxes.size(); i++) {
        for (int j = 0; j < 4; j++) {
            boxes[i][j] = std::round(nms_result[i][j]);

        }
        boxes[i][0] = std::max(0, boxes[i][0]); 
        boxes[i][0] = std::min(img_.cols, boxes[i][0]);
        boxes[i][2] = std::max(0, boxes[i][2]); 
        boxes[i][2] = std::min(img_.cols, boxes[i][2]);

        boxes[i][1] = std::max(0, boxes[i][1]); 
        boxes[i][1] = std::min(img_.rows, boxes[i][1]);
        boxes[i][3] = std::max(0, boxes[i][3]); 
        boxes[i][3] = std::min(img_.rows, boxes[i][3]);
        int c = static_cast<int>(nms_result[i][5]);
        cv::Scalar color = colors[c % colors.size()];
        cv::rectangle(img_, cv::Point(boxes[i][0], boxes[i][1]), cv::Point(boxes[i][2], boxes[i][3]), color, lw, cv::LINE_AA);
        char label[50];
        sprintf(label, "%s %.2f", classes[c].c_str(), nms_result[i][4]);
        cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, tf, &baseLine);
        int h = textSize.height; int w = textSize.width;
        bool outside = (boxes[i][1] - h) >= 3;
        int p2_0 = boxes[i][0] + w;
        int p2_1 = outside ? (boxes[i][1] - h - 3) : (boxes[i][1] + h + 3);
        cv::rectangle(img_, cv::Point(boxes[i][0], boxes[i][1]), cv::Point(p2_0, p2_1), color, cv::FILLED, cv::LINE_AA);
        // std::string label = classes[c] + " " + nms_result[i][4];
        int p1_y = outside ? (boxes[i][1] - 2) : (boxes[i][1] + h + 2);
        cv::putText(img_,
                label,
                cv::Point(boxes[i][0], p1_y),
                cv::FONT_HERSHEY_SIMPLEX,
                lw / 3,
                cv::Scalar(255, 255, 255),
                tf,
                cv::LINE_AA);
        // printf("%d, %d, %d, %d, (%s)\n", boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], label);

    }

    bool sucess = cv::imwrite(mParams.output, img_);
    if (sucess) {
        printf("img save to %s\n", mParams.output.c_str());
    } else {
        printf("img save failed\n");
    }

    return sucess;
}

bool SampleOnnxMNIST::NMS(std::vector<std::vector<float>>& output, float conf_thres, float iou_thres, int max_det, std::vector<std::vector<float>>& nms_result) {
    int outputH = mOutputDims.d[1];
    int outputW = mOutputDims.d[2];

    int nm = 0;
    int bs = mOutputDims.d[0];
    int nc = mOutputDims.d[2] - nm - 5;
    int mi = 5 + nc;
    
    int max_wh = 7680;  // (pixels) maximum box width and height
    int max_nms = 30000;  // maximum number of boxes into torchvision.ops.nms()
    float time_limit = 0.5 + 0.05 * bs;  // seconds to quit after
    bool redundant = true;  // require redundant detections
    // printf("bs: %d, nc: %d, time_limit: %f\n", bs, nc, time_limit);
    std::vector<std::vector<float>> x;
    for (int i = 0; i < outputH; i++) {
        if (output[i][4] > conf_thres) {
            x.emplace_back(output[i]);
        }
    }
    if (x.empty() || x[0].empty()) {
        printf("x.size: (%d, %d)\n", x.size(), x[0].size());
        return false;
    }
    // printf("x[50][5]: %f, x[50][4]: %f\n", x[50][5], x[50][4]);
    std::vector<float> conf(x.size(), 0);
    std::vector<int> arg_max_col(x.size(), -1);
    std::vector<std::vector<float>> box(x.size(), std::vector<float>(4, 0));
    for (int i = 0; i < x.size(); i++) {
        for (int j = 5; j < x[0].size(); j++) {
            x[i][j] *= x[i][4];
            if (x[i][j] > conf[i]) {
                conf[i] = x[i][j];
                arg_max_col[i] = j - 5;
            }
        }
        box[i][0] = x[i][0] - x[i][2] / 2;
        box[i][1] = x[i][1] - x[i][3] / 2;
        box[i][2] = x[i][0] + x[i][2] / 2;
        box[i][3] = x[i][1] + x[i][3] / 2;
        // printf("%f, %f, %f, %f,", box[i][0], box[i][1], box[i][2], box[i][3]);
    }
    std::vector<std::vector<float>> cat_dims;
    for (int i = 0; i < x.size(); i++) {
        if (conf[i] > conf_thres) {
            std::vector<float> tmp = box[i];
            tmp.push_back(conf[i]);
            tmp.push_back(arg_max_col[i]);
            cat_dims.emplace_back(tmp);
        }
    }
    // printf("\ncat_dims.size: %d, cat_dims[0].size: %d", cat_dims.size(), cat_dims[0].size());
    if (cat_dims.size() == 0) {
        sample::gLogError << "no boxes" << std::endl;
        return false;
    }
    std::vector<int> indices(cat_dims.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&cat_dims](int a, int b) { return cat_dims[a][4] > cat_dims[b][4]; });
    if (indices.size() > max_nms)
        indices.resize(max_nms);
    std::vector<std::vector<float>> result;
    for (const auto& index : indices) {
        result.emplace_back(cat_dims[index]);
    }
    std::vector<float> c(result.size(), 0);
    for (int i = 0; i < c.size(); i++) {
        c[i] = result[i][5] * max_wh;
    }
    std::vector<std::vector<float>> boxes(result.size(), std::vector<float>(4, 0));
    for (int i = 0; i < boxes.size(); i++) {
        for (int j = 0; j < boxes[i].size(); j++) {
            boxes[i][j] = result[i][j] + c[i];
        }
    }
    std::vector<float> scores(result.size(), 0);
    for (int i = 0; i < scores.size(); i++) {
        scores[i] = result[i][4];
    }
    // printf("\n");
    std::vector<cv::Rect> rect_boxes;
    indices.clear();
    int x1, x2, y1, y2;
    for (int i = 0; i < boxes.size(); i++) {
        x1 = static_cast<int>(boxes[i][0]);
        y1 = static_cast<int>(boxes[i][1]);
        x2 = static_cast<int>(boxes[i][2]);
        y2 = static_cast<int>(boxes[i][3]);
        cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
        rect_boxes.push_back(rect);
    }
    cv::dnn::NMSBoxes(rect_boxes, scores, /*score_threshold=*/conf_thres, /*nms_threshold=*/iou_thres, indices);
    nms_result.clear();
    for (int index : indices) {
        cv::Rect box = rect_boxes[index];
        nms_result.emplace_back(result[index]);
    }
    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
        params.dataDirs.push_back("./");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "yolov5s.onnx";
    params.inputTensorNames.push_back("images");
    params.outputTensorNames.push_back("output0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.saveEngine = args.saveEngine;
    params.loadEngine = args.loadEngine;
    params.source = args.source;
    params.output = args.output;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    // std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleOnnxMNIST sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for YOLO V5" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    bool build_ret = sample.build();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    sample::gLogInfo << "engine build in: " << duration.count() << " ms." << std::endl;
    if (!build_ret)
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    start = std::chrono::high_resolution_clock::now();
    bool infer_ret = sample.infer();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    sample::gLogInfo << "total engine process time: " << duration.count() << " ms." << std::endl;
    if (!infer_ret)
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
