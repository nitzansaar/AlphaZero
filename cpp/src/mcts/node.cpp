#include "../../include/mcts/node.hpp"
#include <limits>
#include <cmath>

namespace mcts {

MCTSNode::MCTSNode(float prior_prob)
    : prior_prob_(prior_prob), visit_count_(0), total_value_(0.0f) {}

float MCTSNode::value() const {
    if (visit_count_ == 0) {
        return 0.0f;
    }
    return total_value_ / visit_count_;
}

void MCTSNode::update(float value) {
    visit_count_++;
    total_value_ += value;
}

void MCTSNode::add_child(int action, std::unique_ptr<MCTSNode> child) {
    children_[action] = std::move(child);
}

MCTSNode* MCTSNode::get_child(int action) {
    auto it = children_.find(action);
    if (it != children_.end()) {
        return it->second.get();
    }
    return nullptr;
}

float MCTSNode::compute_ucb(float c_puct, int parent_visits) const {
    // UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    // Q(s,a) is the average value (exploitation)
    // The second term is the exploration bonus (higher for unvisited nodes)
    float q_value = value();
    float u_value = c_puct * prior_prob_ * std::sqrt(static_cast<float>(parent_visits)) / (1.0f + visit_count_);
    return q_value + u_value;
}

int MCTSNode::select_child(float c_puct) const {
    float best_score = -std::numeric_limits<float>::infinity();
    int best_action = -1;

    // Parent visit count for exploration term
    int parent_visits = std::max(1, visit_count_);

    for (const auto& [action, child] : children_) {
        float score = child->compute_ucb(c_puct, parent_visits);

        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }

    return best_action;
}

} // namespace mcts
