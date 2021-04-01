#pragma once

#include <sdm/types.hpp>
#include <sdm/core/states.hpp>
#include <sdm/utils/decision_rules/det_decision_rule.hpp>

namespace sdm
{

    class DiscreteMDP;
    class DiscreteMMDP;
    class DiscretePOMDP;
    class DiscreteDecPOMDP;

    template <typename TBelief, typename TAction, typename TObservation>
    class BeliefMDP;

    template <typename oState, typename oAction>
    class OccupancyMDP;

    template <typename oState, typename oAction>
    class SerializedMDP;

    template <typename oState, typename oAction>
    class SerializedOccupancyMDP;

    /**
     * @brief Allows developers to get access to the type of underlying problem that is solved when dealing with a kind of state and action in HSVI.
     * Usage Example : `WorldType<BeliefState, number>::type` will return the type `DiscretePOMDP`. 
     * 
     * @tparam TState the state type
     * @tparam TAction the action type
     */
    template <typename TState, typename TAction = void>
    struct WorldType
    {
        // Return an instance of solvable by hsvi problem given the state type and the action type
        template <bool is_mdp = std::is_same<TState, number>::value>
        static std::enable_if_t<is_mdp, DiscreteMDP>
        getWorldType();

        template <bool is_belief_mdp = std::is_same<TState, BeliefState>::value>
        static std::enable_if_t<is_belief_mdp, BeliefMDP<BeliefState, number, number>>
        getWorldType();

        template <bool is_serialized_mdp = std::is_same<TState, SerializedState>::value>
        static std::enable_if_t<is_serialized_mdp, SerializedMDP<SerializedState, number>>
        getWorldType();

        template <bool is_serialized_occupancy_mdp = std::is_any<TState, SerializedOccupancyState<>>::value>
        static std::enable_if_t<is_serialized_occupancy_mdp, SerializedOccupancyMDP<SerializedOccupancyState<>, DeterministicDecisionRule<HistoryTree_p<number>, number>>>
        getWorldType();

        template <bool is_occupancy_mdp = std::is_same<TState, OccupancyState<>>::value>
        static std::enable_if_t<is_occupancy_mdp, OccupancyMDP<OccupancyState<>, Joint<DeterministicDecisionRule<HistoryTree_p<number>, number>>>>
        getWorldType();

        // Return an instance of the underlying problem that corresponds to the problems solved
        template <bool is_solving_mdp = std::is_same<TState, number>::value>
        static std::enable_if_t<is_solving_mdp, DiscreteMDP>
        getUnderlyingProblem();

        template <bool is_solving_mmdp = std::is_same<TState, SerializedState>::value>
        static std::enable_if_t<is_solving_mmdp, DiscreteMMDP>
        getUnderlyingProblem();

        template <bool is_solving_pomdp = std::is_same<TState, BeliefState>::value>
        static std::enable_if_t<is_solving_pomdp, DiscretePOMDP>
        getUnderlyingProblem();
        
        template <bool is_solving_decpomdp = std::is_any<TState, OccupancyState<number, JointHistoryTree_p<number>>, SerializedOccupancyState<SerializedState, JointHistoryTree_p<number>>>::value>
        static std::enable_if_t<is_solving_decpomdp, DiscreteDecPOMDP>
        getUnderlyingProblem();

        using type = decltype(WorldType<TState, TAction>::getWorldType());
        using underlying_problem_type = decltype(WorldType<TState, TAction>::getUnderlyingProblem());
    };
} // namespace sdm