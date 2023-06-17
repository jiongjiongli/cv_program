// https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#perform_inference_c

// Deserializing a Plan

// 1. Create an instance of the Runtime interface.
IRuntime* runtime = createInferRuntime(logger);

// 2. Read the model into a buffer

// 3. Deserialize it to obtain an engine.
ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize);

// Performing Inference

// 1. Create a context.
IExecutionContext *context = engine->createExecutionContext();

// 2. Pass TensorRT buffers for input and output.
context->setTensorAddress(INPUT_NAME, inputBuffer);
context->setTensorAddress(OUTPUT_NAME, outputBuffer);

// 3. Start inference using a CUDA stream.
context->enqueueV3(stream);

// Infer
https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#run-engine-c

// 1. Deserialize the TensorRT engine from a file. The file contents are read into a buffer and deserialized in-memory
std::vector<char> engineData(fsize);
engineFile.read(engineData.data(), fsize);

std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};

std::unique_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));

// 2. A TensorRT execution context encapsulates execution state such as persistent device memory for holding intermediate activation tensors during inference.
// Since the segmentation model was built with dynamic shapes enabled, the shape of the input must be specified for inference execution. The network output shape may be queried to determine the corresponding dimensions of the output buffer.
auto input_idx = mEngine->getBindingIndex("input");
assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
auto input_dims = nvinfer1::Dims4{1, 3 /* channels */, height, width};
context->setBindingDimensions(input_idx, input_dims);
auto input_size = util::getMemorySize(input_dims, sizeof(float));
auto output_idx = mEngine->getBindingIndex("output");
assert(mEngine->getBindingDataType(output_idx) == nvinfer1::DataType::kINT32);
auto output_dims = context->getBindingDimensions(output_idx);
auto output_size = util::getMemorySize(output_dims, sizeof(int32_t));

// 3. In preparation for inference, CUDA device memory is allocated for all inputs and outputs, image data is processed and copied into input memory, and a list of engine bindings is generated.
oid* input_mem{nullptr};
cudaMalloc(&input_mem, input_size);
void* output_mem{nullptr};
cudaMalloc(&output_mem, output_size);
const std::vector<float> mean{0.485f, 0.456f, 0.406f};
const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
auto input_image{util::RGBImageReader(input_filename, input_dims, mean, stddev)};
input_image.read();
auto input_buffer = input_image.process();
cudaMemcpyAsync(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice, stream);

// 4. Inference execution is kicked off using the context’s executeV2 or enqueueV2 methods. After the execution is complete, we copy the results back to a host buffer and release all device memory allocations.
void* bindings[] = {input_mem, output_mem};
bool status = context->enqueueV2(bindings, stream, nullptr);
auto output_buffer = std::unique_ptr<int>{new int[output_size]};
cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);

cudaFree(input_mem);
cudaFree(output_mem);
