#pragma once

/*
 * nn_inference.h — LibTorch TorchScript model inference for chess.
 *
 * Loads a TorchScript model produced by export_model.py and exposes a batched
 * eval() method compatible with the NNEvalFn typedef in mcts.h.
 *
 * Usage in the selfplay binary:
 *   NNInference nn("output_chess/models/5/model_ts.pt", use_cuda);
 *   static thread_local NNInference *tl_nn = &nn;
 *   auto cb = [](const float *p, int bs, float *v, float *pol){
 *       tl_nn->eval(p, bs, v, pol);
 *   };
 *   mcts_simulate(&pool, cb, 400, 32, true);
 */

#include "mcts.h"           /* NNEvalFn, CHESS_NUM_PLANES, CHESS_ACTION_SIZE */

#include <torch/script.h>
#include <string>

class NNInference {
public:
    explicit NNInference(const std::string &model_path, bool use_cuda = false);

    /*
     * Batched evaluation.  Signature matches NNEvalFn.
     *   planes   : batch_size × CHESS_NUM_PLANES × 64 floats, row-major
     *   values   : [out] batch_size scalars in [-1, 1]   (value head, tanh)
     *   policies : [out] batch_size × CHESS_ACTION_SIZE softmax probabilities
     *              (softmax applied here; the network outputs raw logits)
     */
    void eval(const float *planes, int batch_size,
              float *values, float *policies);

    bool on_cuda() const { return device_.is_cuda(); }

private:
    torch::jit::script::Module module_;
    torch::Device              device_;
};
