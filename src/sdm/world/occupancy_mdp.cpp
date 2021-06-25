#include <sdm/world/occupancy_mdp.hpp>
#include <sdm/core/state/jhistory_tree.hpp>
#include <sdm/core/space/function_space.hpp>
#include <sdm/core/space/multi_discrete_space.hpp>
#include <sdm/core/action/det_decision_rule.hpp>
#include <sdm/core/state/occupancy_state.hpp>
#include <sdm/core/action/joint_det_decision_rule.hpp>

namespace sdm
{

    OccupancyMDP::OccupancyMDP() {}

    OccupancyMDP::OccupancyMDP(std::shared_ptr<MPOMDPInterface> underlying_dpomdp, number max_history_length)
        : BeliefMDP(underlying_dpomdp)
    {
        // Initialize history
        this->initial_history_ = std::make_shared<JointHistoryTree>(this->getUnderlyingMPOMDP()->getNumAgents(), (max_history_length > 0) ? max_history_length : -1);

        // Initialize empty state
        auto initial_state = std::make_shared<OccupancyState>(this->getUnderlyingMPOMDP()->getNumAgents());
        initial_state->setProbability(this->initial_history_->toJointHistory(), this->initial_state_->toBelief(), 1);

        this->initial_state_ = initial_state;
        this->initial_state_->toOccupancyState()->finalize();
        this->initial_state_->toOccupancyState()->setFullyUncompressedOccupancy(this->initial_state_->toOccupancyState());
        this->initial_state_->toOccupancyState()->setOneStepUncompressedOccupancy(this->initial_state_->toOccupancyState());
    }

    std::tuple<std::shared_ptr<State>, std::vector<double>, bool> OccupancyMDP::step(std::shared_ptr<Action> joint_idr)
    {
        // Select joint action
        // const auto &jaction = joint_idr->toDecisionRule()->act(this->current_state_->toOccupancyState()->getJointLabels(this->current_history_->toJointHistory()->getIndividualHistories()));

        // // Do a step on the DecPOMDP and get next observation and rewards
        // const auto &[next_obs, rewards, done] = this->getUnderlyingProblem()->step(jaction);

        // // Expand the current history
        // this->current_history_ = this->current_history_->expand(next_obs);

        // // Compute the next compressed occupancy state
        // *this->current_state_ = this->nextState(*this->current_state_, joint_idr);

        // // return the new occupancy state and the perceived rewards
        // return std::make_tuple(*this->current_state_, rewards, done);
    }

    std::shared_ptr<Space> OccupancyMDP::getActionSpaceAt(const std::shared_ptr<State> &ostate, number t)
    {
        std::vector<std::shared_ptr<Space>> vector_indiv_space;
        for (int agent_id = 0; agent_id < this->getUnderlyingProblem()->getNumAgents(); agent_id++)
        {
            // Get history space of agent i
            auto set_history_i = ostate->toOccupancyState()->getIndividualHistories(agent_id);
            auto history_space_i = std::make_shared<DiscreteSpace>(sdm::tools::set2vector(set_history_i));
            // Get action space of agent i
            auto action_space_i = std::static_pointer_cast<MultiDiscreteSpace>(this->getUnderlyingProblem()->getActionSpace(t))->get(agent_id);
            // Add individual decision rule space of agent i
            vector_indiv_space.push_back(std::make_shared<FunctionSpace<DeterministicDecisionRule>>(history_space_i, action_space_i, false));
        }

        // Now we can return a discrete space of all joint decision rules
        std::shared_ptr<Space> null_space = std::make_shared<DiscreteSpace>(std::vector<std::shared_ptr<Item>>{std::make_shared<DiscreteState>(3)});
        std::shared_ptr<Space> joint_decision_rule_space = std::make_shared<MultiDiscreteSpace>(vector_indiv_space, false);

        auto function_space = std::make_shared<FunctionSpace<JointDeterministicDecisionRule>>(null_space, joint_decision_rule_space, false);
        return function_space;
    }

    std::shared_ptr<OccupancyStateInterface> OccupancyMDP::nextState(const std::shared_ptr<State> &state_tmp, const std::shared_ptr<Action> &action_tmp, number t, const std::shared_ptr<HSVI> &, bool compression) const
    {
        try
        {
            auto occupancy_state = state_tmp->toOccupancyState();
            auto joint_idr = action_tmp->toDecisionRule();
            // The new compressed occupancy state
            std::shared_ptr<OccupancyStateInterface> new_compressed_occupancy_state;
            // The new fully uncompressed occupancy state
            std::shared_ptr<OccupancyStateInterface> new_fully_uncompressed_occupancy_state = std::make_shared<OccupancyState>(this->getUnderlyingMPOMDP()->getNumAgents());
            // The new one step left occupancy state
            std::shared_ptr<OccupancyStateInterface> new_one_step_left_compressed_occupancy_state = std::make_shared<OccupancyState>(this->getUnderlyingMPOMDP()->getNumAgents());

            for (const auto &joint_history : occupancy_state->getFullyUncompressedOccupancy()->getJointHistories())
            {
                auto joint_action = joint_idr->act(occupancy_state->getJointLabels(joint_history->getIndividualHistories()).toJoint<State>());

                for (const auto &belief : occupancy_state->getBeliefsAt(joint_history))
                {
                    for (auto &joint_observation : this->getUnderlyingMPOMDP()->getReachableObservations(belief, joint_action, belief, t))
                    {
                        // p(o') = p(o) * p(z | b, a)
                        double proba_next_history = occupancy_state->getProbability(joint_history, belief) * BeliefMDP::getObservationProbability(belief, joint_action, nullptr, joint_observation, t);

                        if (proba_next_history > 0)
                        {
                            auto next_joint_history = joint_history->expand(joint_observation);
                            auto next_belief = BeliefMDP::nextState(belief, joint_action, joint_observation, t);

                            // Build fully uncompressed occupancy state
                            new_fully_uncompressed_occupancy_state->setProbability(next_joint_history->toJointHistory(), next_belief->toBelief(), proba_next_history);

                            // Build one step left uncompressed occupancy state
                            auto compressed_joint_history = occupancy_state->getCompressedJointHistory(joint_history);

                            auto next_compressed_joint_history = compressed_joint_history->expand(joint_observation);
                            new_one_step_left_compressed_occupancy_state->addProbability(next_compressed_joint_history->toJointHistory(), next_belief->toBelief(), proba_next_history);

                            // Update next history labels
                            new_one_step_left_compressed_occupancy_state->updateJointLabels(next_joint_history->toJointHistory()->getIndividualHistories(), next_compressed_joint_history->toJointHistory()->getIndividualHistories());
                        }
                    }
                }
            }

            // Finalize the one step left compressed occupancy state
            new_one_step_left_compressed_occupancy_state->finalize();

            if (compression)
            {
                // Compress the occupancy state
                new_compressed_occupancy_state = new_one_step_left_compressed_occupancy_state->compress();
                new_compressed_occupancy_state->setFullyUncompressedOccupancy(new_fully_uncompressed_occupancy_state);
                new_compressed_occupancy_state->setOneStepUncompressedOccupancy(new_one_step_left_compressed_occupancy_state);

                return new_compressed_occupancy_state;
            }

            new_one_step_left_compressed_occupancy_state->setFullyUncompressedOccupancy(new_fully_uncompressed_occupancy_state);
            new_one_step_left_compressed_occupancy_state->setOneStepUncompressedOccupancy(new_one_step_left_compressed_occupancy_state);
            return new_one_step_left_compressed_occupancy_state;
        }
        catch (const std::exception &exc)
        {
            std::cout << state_tmp->str() << std::endl;
            std::cout << action_tmp->str() << std::endl;
            // catch anything thrown within try block that derives from std::exception
            std::cerr << "OccupancyMDP::nextState(..) exception caught: " << exc.what() << std::endl;
            exit(-1);
        }
    }

    std::shared_ptr<State> OccupancyMDP::nextState(const std::shared_ptr<State> &ostate, const std::shared_ptr<Action> &joint_idr, number h, const std::shared_ptr<HSVI> &hsvi) const
    {
        return this->nextState(ostate, joint_idr, h, hsvi, true);
    }

    double OccupancyMDP::getReward(const std::shared_ptr<State> &occupancy_state, const std::shared_ptr<Action> &decision_rule, number t) const
    {
        double reward = 0;
        for (const auto &joint_history : occupancy_state->toOccupancyState()->getJointHistories())
        {
            for (const auto &belief : occupancy_state->toOccupancyState()->getBeliefsAt(joint_history))
            {
                // Get list of individual history labels
                auto joint_hist = occupancy_state->toOccupancyState()->getJointLabels(joint_history->getIndividualHistories()).toJoint<State>();

                // Get selected joint action
                auto action = std::static_pointer_cast<JointDeterministicDecisionRule>(decision_rule)->act(joint_hist);

                // Transform selected joint action into joint action address
                auto joint_action = std::static_pointer_cast<Joint<std::shared_ptr<Action>>>(action);
                auto joint_action_address = std::static_pointer_cast<MultiDiscreteSpace>(this->getUnderlyingProblem()->getActionSpace(t))->getItemAddress(*joint_action->toJoint<Item>());

                reward += occupancy_state->toOccupancyState()->getProbability(joint_history, belief) * BeliefMDP::getReward(belief, joint_action_address->toAction(), t);
                // for (const auto &state : ostate->getStatesAt(joint_history))
                // {
                //     reward += ostate->toOccupancyState()->getProbability({state, joint_history}) * this->getUnderlyingMPOMDP()->getReward(state, jaction);
                // }
            }
        }
        return reward;
    }

    double OccupancyMDP::getExpectedNextValue(const std::shared_ptr<ValueFunction> &value_function, const std::shared_ptr<State> &occupancy_state, const std::shared_ptr<Action> &joint_decision_rule, number t) const
    {
        return value_function->getValueAt(this->nextState(occupancy_state, joint_decision_rule, t), t + 1);

    }

    std::shared_ptr<MPOMDPInterface> OccupancyMDP::getUnderlyingMPOMDP() const
    {
        return std::dynamic_pointer_cast<MPOMDPInterface>(this->getUnderlyingMDP());
    }
} // namespace sdm
