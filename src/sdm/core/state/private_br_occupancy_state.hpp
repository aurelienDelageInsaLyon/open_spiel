#pragma once

#include <sdm/core/state/occupancy_state.hpp>
#include <sdm/world/base/pomdp_interface.hpp>
#include <sdm/core/action/decision_rule.hpp>
#include <sdm/core/action/stochastic_decision_rule.hpp>

namespace sdm
{

    /**
     * @brief A private occupancy state is an occupancy state (i.e. \$p(s_t, \\theta^{-i}_{t} \\mid \\iota_t, \\theta_t^i)\$ ).
     *
     */
    class PrivateBrOccupancyState : public OccupancyState
    {
    public:
        static double PRECISION_COMPRESSION;

        PrivateBrOccupancyState();
        PrivateBrOccupancyState(number num_agents, number h, StochasticDecisionRule dr);
        PrivateBrOccupancyState(number agent_id, number num_agents, number h, StochasticDecisionRule dr);
        PrivateBrOccupancyState(number agent_id, number num_agents, number h);
        PrivateBrOccupancyState(const PrivateBrOccupancyState &);
        PrivateBrOccupancyState(const OccupancyState &);
        PrivateBrOccupancyState(const OccupancyState &, number id, StochasticDecisionRule dr);
        
        virtual std::shared_ptr<OccupancyState> make(number h);

        virtual double getReward(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, number t);

        double reward=0;
            std::shared_ptr<sdm::JointAction> getActionPointer(std::shared_ptr<sdm::Action> a, std::shared_ptr<sdm::Action> b,
    std::shared_ptr<POMDPInterface> pomdp, number agent_id_) const;
    std::shared_ptr<sdm::Observation> getObservationPointer(std::shared_ptr<sdm::Observation> a, std::shared_ptr<sdm::Observation> b,
    std::shared_ptr<POMDPInterface> pomdp, number agent_id_) const;

    void updateOccupancyStateProba(const std::shared_ptr<OccupancyStateInterface> &occupancy_state, const std::shared_ptr<JointHistoryInterface> &joint_history, const std::shared_ptr<BeliefInterface> &belief, double probability);

    /*
        std::shared_ptr<sdm::Action> getIndivActionPointer(std::shared_ptr<sdm::Action> a,
    std::shared_ptr<POMDPInterface> pomdp) const;
    std::shared_ptr<sdm::Observation> getIndivObservationPointer(std::shared_ptr<sdm::Observation> a,
    std::shared_ptr<POMDPInterface> pomdp) const;
    */
        /**
         * @brief Get the id of the agent that related to this occupancy state.
         *
         * @return number the agent id
         */
        number getAgentId() const;

        /**
         * @brief Get the partial joint history $\theta^{-i}$ corresponding to the joint history $\theta$
         *
         * @return the corresponding partial history
         */
        const std::vector<std::shared_ptr<HistoryInterface>> &getPartialJointHistory(const std::shared_ptr<JointHistoryInterface> &) const;

        /**
         * @brief Get the full joint history $\theta$ corresponding to the partial joint history $\theta^{-i}$
         *
         * @return the joint history
         */
        std::shared_ptr<JointHistoryInterface> getJointHistoryFromPartial(const std::vector<std::shared_ptr<HistoryInterface>> &) const;

        void finalize();
        void finalize(bool do_compression);

        /**
         * @brief Check the equivalence between two private occupancy states.
         *
         * @return true if private occupancy states are equivalent (given precision PRECISION_COMPRESSION ).
         * @return false if private occupancy states are not equivalent
         */
        bool check_equivalence(const PrivateBrOccupancyState &) const;

        std::string str() const;

        std::shared_ptr<Action> applyDR(const std::shared_ptr<DecisionRule> &dr, const std::shared_ptr<HistoryInterface> &individual_history) const;
        
        std::shared_ptr<Action> applyStochasticDR(const StochasticDecisionRule dr, const std::shared_ptr<HistoryInterface> &individual_history) const;

        /*
        Pair<std::shared_ptr<BeliefInterface>, double> getNewBelief(const std::shared_ptr<BeliefInterface> &belief, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, const std::shared_ptr<Observation> & jobs, number t, const std::shared_ptr<MDPInterface> &mdp) const;
        */

        Pair<std::shared_ptr<State>, double> computeNext(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, number t);

        StochasticDecisionRule strategyOther;

        double norm_1_other(const std::shared_ptr<OccupancyState>  &other);

        /** @brief The agent's identifier */
        number agent_id_;

    protected:
        /**
         * @brief Get the partial joint history $\theta^{-i}$ corresponding to the joint history $\theta$
         *
         * @return the corresponding partial history
         */
        std::vector<std::shared_ptr<HistoryInterface>> getPartialJointHistory(const std::vector<std::shared_ptr<HistoryInterface>> &) const;


        std::unordered_map<std::shared_ptr<JointHistoryInterface>, Joint<std::shared_ptr<HistoryInterface>>> map_jhist_to_partial;
        std::unordered_map<Joint<std::shared_ptr<HistoryInterface>>, std::shared_ptr<JointHistoryInterface>> map_partial_to_jhist;
    };
} // namespace sdm

DEFINE_STD_HASH(sdm::PrivateBrOccupancyState, sdm::OccupancyState::PRECISION);
