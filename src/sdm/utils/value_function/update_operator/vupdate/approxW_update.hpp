
#pragma once

#include <sdm/utils/linear_algebra/hyperplane/hyperplane.hpp>
#include <sdm/utils/value_function/update_operator/vupdate_operator.hpp>
#include <sdm/utils/value_function/vfunction/partial_Q_value_function.hpp>

//TODO importer FactoredOccupancyState

namespace sdm
{
    namespace update
    {
        class UpdateW : public UpdateOperator<partialQValueFunction>
        {
        public:
            UpdateW(const std::shared_ptr<ValueFunctionInterface> &value_function);

            void update(const std::shared_ptr<State> &state, number t);
            void update(const std::shared_ptr<State> &state, const std::shared_ptr<Action> &action, number t);
            void update(const std::shared_ptr<State> &state, double new_value, number t){}
            void update(const std::shared_ptr<State> &state, number t,const std::shared_ptr<State> &lastState, const std::shared_ptr<StochasticDecisionRule> & dr);

            void updateLastStep(const std::shared_ptr<State> &state, number t,const std::shared_ptr<State> &lastState,const std::shared_ptr<StochasticDecisionRule> & lastDr,std::map<std::shared_ptr<HistoryInterface>,double> valuesUpb,const std::shared_ptr<StochasticDecisionRule> & dr);

        protected:
            //TODO penser a l'option template. Mais pour cela, il faut que toutes les deux fonctions aient les familles d'arguments.
            std::shared_ptr<Hyperplane> computeNewHyperplane(const std::shared_ptr<BeliefInterface> &belief_state, number t);
            std::shared_ptr<Hyperplane> computeNewHyperplane(const std::shared_ptr<OccupancyStateInterface> &occupancy_state, const std::shared_ptr<Action> &decision_rule, number t);
        };
    }
}