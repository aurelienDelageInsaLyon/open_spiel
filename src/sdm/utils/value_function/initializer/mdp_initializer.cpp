
#include <sdm/algorithms/planning/hsvi.hpp>
#include <sdm/utils/value_function/initializer/mdp_initializer.hpp>

#include <sdm/utils/value_function/vfunction/tabular_value_function.hpp>
#include <sdm/utils/value_function/action_selection/exhaustive_action_selection.hpp>
#include <sdm/utils/value_function/update_operator/vupdate/tabular_update.hpp>

#include <sdm/utils/value_function/initializer/state_2_occupancy_vf.hpp>
#include <sdm/world/solvable_by_mdp.hpp>
#include <sdm/algorithms/planning/vi.hpp>

namespace sdm
{
    namespace algo
    {
        // std::shared_ptr<sdm::HSVI> makeHSVI(std::shared_ptr<SolvableByHSVI> problem, std::string upper_bound, std::string lower_bound, std::string ub_init_name, std::string lb_init_name, double discount, double error, number horizon, int trials, std::string name, std::string current_type_of_resolution , number BigM , std::string type_sawtooth_linear_programming );

        // std::shared_ptr<sdm::ValueIteration> makeValueIteration(std::shared_ptr<SolvableByHSVI> problem, double error, number horizon);
    }
}
namespace sdm
{
    MDPInitializer::MDPInitializer(std::shared_ptr<SolvableByHSVI> world, std::string algo_name, double error, int trials) : algo_name_(algo_name), error_(error), trials_(trials), world_(world)
    {
    }

    void MDPInitializer::init(std::shared_ptr<ValueFunctionInterface> vf)
    {
        auto value_function = std::dynamic_pointer_cast<ValueFunction>(vf);
        // Get relaxed MDP problem and thgetUnderlyingProbleme underlying problem
        auto mdp = this->world_->getUnderlyingProblem();
        std::shared_ptr<SolvableByHSVI> hsvi_mdp = std::make_shared<SolvableByMDP>(mdp);

        // if (this->algo_name_ == "ValueIteration")
        // {
        //     auto value_iteration = std::make_shared<sdm::ValueIteration>(hsvi_mdp, this->error_, mdp->getHorizon());

        //     value_iteration->initialize();
        //     value_iteration->solve();

        //     vf->initialize(std::make_shared<State2OccupancyValueFunction>(value_iteration->getValueFunction()));
        // }
        // else
        // {
        auto action_tabular = std::make_shared<ExhaustiveActionSelection>(hsvi_mdp);

        auto init_lb = std::make_shared<MinInitializer>(hsvi_mdp);
        auto init_ub = std::make_shared<MaxInitializer>(hsvi_mdp);

        auto lb = std::make_shared<TabularValueFunction>(hsvi_mdp, init_lb, action_tabular, nullptr, false);
        lb->setUpdateOperator(std::make_shared<update::TabularUpdate>(lb));

        auto ub = std::make_shared<TabularValueFunction>(hsvi_mdp, init_ub, action_tabular, nullptr, true);
        ub->setUpdateOperator(std::make_shared<update::TabularUpdate>(ub));

        auto algorithm = std::make_shared<HSVI>(hsvi_mdp, lb, ub,this->error_, 100000, "MDP_Initialisation");

        algorithm->initialize();

        for (const auto &element : *mdp->getStateSpace(0))
        {
            auto state = element->toState();
            hsvi_mdp->setInitialState(state);
            algorithm->solve();
        }

        auto ubound = algorithm->getUpperBound();
        value_function->setInitFunction(std::make_shared<State2OccupancyValueFunction>(ubound));
        // }
        // Set the function that will be used to get interactively upper bounds
    }
} // namespace sdm
