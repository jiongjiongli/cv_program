

// https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#c_topics

#include “NvInfer.h”

using namespace nvinfer1;

// The Build Phase
// 1. Instantiate the ILogger interface.
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

// 2. Create an instance of the builder
IBuilder* builder = createInferBuilder(logger);

// 3. Create a network definition. It's the first step in optimizing a model.
uint32_t flag = 1U << static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

INetworkDefinition* network = builder->createNetworkV2(flag);


#include "NvOnnxParser.h"

using namespace nvonnxparser;

// 4. Create an ONNX parser to populate the network definition from the ONNX representation.
IParser* parser = createParser(*network, logger);

// 5. Read the model file and process any errors.
parser->parseFromFile(modelFile,
    static_cast<int32_t>(ILogger::Severity::kWARNING));

for (int32_t i = 0; i < parser.getNbErrors(); ++i)
{
    std::cout << parser->getError(i)->desc() << std::endl;
}

// 6. Create a build configuration specifying how TensorRT should optimize the model.
// This interface has many properties that you can set in order to control how TensorRT optimizes the network.
IBuilderConfig* config = builder->createBuilderConfig();
config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);

// 7. Build serialized engine.
IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

// 8. Since the serialized engine contains the necessary copies of the weights, the parser, network definition, builder configuration and builder are no longer necessary and may be safely deleted:
delete parser;
delete network;
delete config;
delete builder;

