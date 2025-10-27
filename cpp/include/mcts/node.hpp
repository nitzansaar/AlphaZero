#pragma once
#include <memory>
#include <unordered_map>
#include <cmath>

namespace mcts {

class MCTSNode {
public:
    explicit MCTSNode(float prior_prob);

    // UCB selection
    int select_child(float c_puct) const;
    float compute_ucb(float c_puct, int parent_visits) const;

    // Statistics
    float value() const;
    void update(float value);

    // Tree structure
    void add_child(int action, std::unique_ptr<MCTSNode> child);
    MCTSNode* get_child(int action);
    bool has_children() const { return !children_.empty(); }

    // Getters
    int visit_count() const { return visit_count_; }
    float prior_prob() const { return prior_prob_; }
    float total_value() const { return total_value_; }

    const std::unordered_map<int, std::unique_ptr<MCTSNode>>& children() const {
        return children_;
    }

private:
    float prior_prob_;
    int visit_count_;
    float total_value_;
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children_;
};

} // namespace mcts
