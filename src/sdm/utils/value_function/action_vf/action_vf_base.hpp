#pragma once

#include <sdm/utils/value_function/action_vf/action_vf_interface.hpp>
#include <sdm/world/solvable_by_hsvi.hpp>
#include <sdm/utils/value_function/value_function.hpp>

namespace sdm
{
    template <class TData>
    class ActionVFBase : public ActionVFInterface<TData>
    {
    public:
        
        ActionVFBase();
        ActionVFBase(const std::shared_ptr<SolvableByHSVI>& world);
        
        virtual ~ActionVFBase();
        /**
         * @brief Select the best action for a state at a precise time
         * 
         * @param const std::shared_ptr<ValueFunction>& vf : Value function
         * @param const std::shared_ptr<State>& state : current state
         * @param number t : time step
         * @return std::shared_ptr<Action> : Action
         */
        virtual Pair<std::shared_ptr<Action>,TData> selectBestAction(const std::shared_ptr<ValueFunction>& vf, const std::shared_ptr<State>& state, number t) = 0;
    protected : 
        std::shared_ptr<SolvableByHSVI> world_;
    };
}
#include <sdm/utils/value_function/action_vf/action_vf_base.tpp>
