#include "nn_inference.h"

#include <torch/torch.h>
#include <stdexcept>

/* ── Constructor ──────────────────────────────────────────────────────── */

NNInference::NNInference(const std::string &model_path, bool use_cuda)
    : device_(torch::kCPU)
{
    if (use_cuda && torch::cuda::is_available())
        device_ = torch::Device(torch::kCUDA);

    try {
        module_ = torch::jit::load(model_path, device_);
    } catch (const c10::Error &e) {
        throw std::runtime_error(
            std::string("NNInference: failed to load '") + model_path
            + "': " + e.what());
    }

    module_.eval();
}

/* ── eval ─────────────────────────────────────────────────────────────── */

void NNInference::eval(const float *planes, int batch_size,
                       float *values, float *policies)
{
    /*
     * planes layout: batch × CHESS_NUM_PLANES × 64 (flat).
     * NN expects:    batch × CHESS_NUM_PLANES × 8 × 8 (same contiguous memory).
     * from_blob wraps without copying; .to(device_) copies to GPU when needed.
     */
    torch::Tensor input = torch::from_blob(
        const_cast<float *>(planes),
        {batch_size, CHESS_NUM_PLANES, 8, 8},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).to(device_);

    torch::NoGradGuard no_grad;

    std::vector<torch::jit::IValue> inputs = {input};
    auto out_tuple = module_.forward(inputs).toTuple();

    /* Value head: (batch, 1) → squeeze to (batch,), move to CPU. */
    torch::Tensor val_t = out_tuple->elements()[0].toTensor().squeeze(1).cpu();

    /* Policy head: (batch, ACTION_SIZE) raw logits → softmax → CPU. */
    torch::Tensor pol_t = torch::softmax(
        out_tuple->elements()[1].toTensor(), /*dim=*/1
    ).cpu().contiguous();

    const float *val_ptr = val_t.contiguous().data_ptr<float>();
    const float *pol_ptr = pol_t.data_ptr<float>();

    for (int b = 0; b < batch_size; b++)
        values[b] = val_ptr[b];

    for (int i = 0; i < batch_size * CHESS_ACTION_SIZE; i++)
        policies[i] = pol_ptr[i];
}
