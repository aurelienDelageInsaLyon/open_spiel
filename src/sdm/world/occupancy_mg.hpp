#pragma once

#include <sdm/utils/config.hpp>
#include <sdm/world/belief_mdp.hpp>
#include <sdm/world/base/mpomdp_interface.hpp>
#include <sdm/core/state/interface/history_interface.hpp>
#include <sdm/core/state/interface/occupancy_state_interface.hpp>
#include <sdm/core/state/jhistory_tree.hpp>
#include <sdm/core/space/function_space.hpp>
#include <sdm/core/space/multi_discrete_space.hpp>
#include <sdm/core/action/det_decision_rule.hpp>
#include <sdm/core/state/occupancy_state.hpp>
#include <sdm/core/action/joint_det_decision_rule.hpp>
#include <sdm/core/state/occupancy_state_mg.hpp>
#include <sdm/world/occupancy_mdp.hpp>

/**
 * @namespace  sdm
 * @brief Namespace grouping all tools required for sequential decision making.
 */
namespace sdm
{

        /**
         * @brief This class provides a way to transform a Dec-POMDP into an occupancy MDP formalism.
         *
         * This problem reformulation can be used to solve the underlying Dec-POMDP with standard dynamic programming algorithms.
         *
         */
        template <class TOccupancyState = OccupancyStateMG>
        class BaseOccupancyMG : public BaseOccupancyMDP<TOccupancyState>,
                                 public std::enable_shared_from_this<BaseOccupancyMG<TOccupancyState>>
        {
        public:
                BaseOccupancyMG();
                BaseOccupancyMG(Config config);
                BaseOccupancyMG(const std::shared_ptr<MPOMDPInterface> &decpomdp, const number num_player_, Config config);
                BaseOccupancyMG(const std::shared_ptr<MPOMDPInterface> &decpomdp,const number num_player, int memory = -1,
                                 bool store_states = true, bool store_actions = true, int batch_size = 0);

                std::vector<Action> getBaseActions(number agent_id_);

                ~BaseOccupancyMG();
        
                std::shared_ptr<OccupancyStateMG> initial_state_;

                std::shared_ptr<State> getInitialState();
                
                virtual std::shared_ptr<Space> getObservationSpaceAt(const std::shared_ptr<State> &state, const std::shared_ptr<Action> &action, number t);
        };

        using OccupancyMG = BaseOccupancyMG<OccupancyStateMG>;

} // namespace sdm

#include <sdm/world/occupancy_mg.tpp>
