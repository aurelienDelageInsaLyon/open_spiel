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
    class OccupancyStateMG : public OccupancyState
    {
    public:
        static double PRECISION_COMPRESSION;

        OccupancyStateMG();
        OccupancyStateMG(number num_agents, number h);
        OccupancyStateMG(const OccupancyStateMG &);
        OccupancyStateMG(const OccupancyState &);
    
        double norm_1_other(const std::shared_ptr<OccupancyStateMG> & other);

        virtual std::shared_ptr<OccupancyState> make(number h);

        virtual double getReward(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, number t);
        
        /*
        virtual std::vector<Pair<std::shared_ptr<Action>,double>> applyStochasticJointDR(const std::shared_ptr<DecisionRule> &dr, const std::shared_ptr<JointHistoryInterface> &joint_history) const;
        */
       
         Pair<std::shared_ptr<State>, double> computeNext(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, number t);

        };
} // namespace sdm

DEFINE_STD_HASH(sdm::OccupancyStateMG, sdm::OccupancyState::PRECISION);