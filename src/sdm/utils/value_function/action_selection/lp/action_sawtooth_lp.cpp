#ifdef WITH_CPLEX
#include <sdm/utils/value_function/action_selection/lp/action_sawtooth_lp.hpp>
#include <sdm/core/state/interface/occupancy_state_interface.hpp>
#include <sdm/core/state/private_occupancy_state.hpp>
#include <sdm/core/state/interface/serial_interface.hpp>

#include <sdm/world/base/mpomdp_interface.hpp>
#include <sdm/world/occupancy_mdp.hpp>

namespace sdm
{
    ActionSelectionSawtoothLP::ActionSelectionSawtoothLP() {}
    ActionSelectionSawtoothLP::ActionSelectionSawtoothLP(const std::shared_ptr<SolvableByDP> &world, TypeOfResolution current_type_of_resolution, number bigM_value, TypeSawtoothLinearProgram type_of_linear_program) : ActionSelectionBase(world), DecentralizedLP(world), current_type_of_resolution_(current_type_of_resolution), type_of_linear_program_(type_of_linear_program)
    {
        this->bigM_value_ = bigM_value;
    }

    Pair<std::shared_ptr<Action>, double> ActionSelectionSawtoothLP::getGreedyActionAndValue(const std::shared_ptr<ValueFunctionInterface> &vf, const std::shared_ptr<State> &one_step_uncompressed_state, number t)
    {
        this->sawtooth_vf = std::dynamic_pointer_cast<SawtoothValueFunction>(vf);

        auto result = this->createLP(vf, one_step_uncompressed_state->toOccupancyState()->getCompressedOccupancy(), t);
        return result;
    }

    void ActionSelectionSawtoothLP::createObjectiveFunction(const std::shared_ptr<ValueFunctionInterface> &, const std::shared_ptr<State> &, IloNumVarArray &var, IloObjective &obj, number)
    {
        // <! 1.a get variable v
        auto recover = this->getNumber(this->getVarNameWeight(0));
        obj.setLinearCoef(var[recover], 1);
    }

    void ActionSelectionSawtoothLP::createVariables(const std::shared_ptr<ValueFunctionInterface> &, const std::shared_ptr<State> &state, IloEnv &env, IloNumVarArray &var, number &index, number t)
    {

        //<! 0.b Build variables v_0 = objective variable!
        std::string VarName = this->getVarNameWeight(0);
        var.add(IloNumVar(env, -IloInfinity, IloInfinity, VarName.c_str()));
        this->setNumber(VarName, index++);

        //<! Define variables \omega_k(x',o')
        // Go over all Point Set in t+1
        for (const auto &point_k : getSawtoothValueFunction()->getRepresentation(t + 1))
        {
            const auto &next_occupancy_state = point_k.first->toOccupancyState();

            // Go over all Joint History Next
            for (const auto &next_jhistory : next_occupancy_state->getJointHistories())
            {
                // <! \omega_k(x',o')
                VarName = this->getVarNameWeightedStateJointHistory(next_occupancy_state, next_jhistory);
                var.add(IloBoolVar(env, 0, 1, VarName.c_str()));
                this->setNumber(VarName, index++);
            }
        }

        // Create Decentralized Variables
        DecentralizedLP::createVariables(getSawtoothValueFunction(), state, env, var, index, t);
    }

    //<!  Build sawtooth constraints v - \sum_{u} a(u|o) * Q(k,s,o,u,y,z, diff, t  ) + \omega_k(y,<o,z>)*M <= M,  \forall k, y,<o,z>
    //<!  Build sawtooth constraints  Q(k,s,o,u,y,z, diff, t ) = (v_k - V_k) \frac{\sum_{x} s(x,o) * p(x,u,z,y)}}{s_k(y,<o,z>)},  \forall a(u|o)
    void ActionSelectionSawtoothLP::createConstraints(const std::shared_ptr<ValueFunctionInterface> &, const std::shared_ptr<State> &occupancy_state, IloEnv &env, IloModel &model, IloRangeArray &con, IloNumVarArray &var, number &index, number t)
    {
        assert(getSawtoothValueFunction()->getInitFunction() != nullptr);

        auto compressed_occupancy_state = occupancy_state->toOccupancyState();

        bool is_empty = getSawtoothValueFunction()->getRepresentation(t + 1).empty();
        if (is_empty)
        {
            this->createInitialConstraints(compressed_occupancy_state, env, con, var, index, t);
        }
        else
        {
            // Go over all points in the point set at t+1
            for (const auto &point_k : getSawtoothValueFunction()->getRepresentation(t + 1))
            {
                // std::cout << "\n\n-------- Point ----------" << std::endl;
                const auto &next_occupancy_state = point_k.first->toOccupancyState();

                // Go over all Joint History Next
                for (const auto &next_jhistory : next_occupancy_state->getJointHistories())
                {
                    // Determine the next Joint observation thanks to the next joint history
                    auto next_joint_observation = this->determineNextJointObservation(next_jhistory, t);

                    // We search for the joint_history which allow us to obtain the current next_history conditionning to the next joint observation
                    switch (this->current_type_of_resolution_)
                    {
                    case TypeOfResolution::BigM:
                        this->createSawtoothBigM(compressed_occupancy_state, next_occupancy_state, next_jhistory, next_joint_observation, env, con, var, index, t);
                        break;
                    case TypeOfResolution::IloIfThenResolution:
                        this->createSawtoothIloIfThen(compressed_occupancy_state, next_occupancy_state, next_jhistory, next_joint_observation, env, model, var, t);
                        break;
                    }
                }
            }

            this->createOmegaConstraints(env, con, var, index, t);
        }

        DecentralizedLP::createConstraints(getSawtoothValueFunction(), occupancy_state, env, model, con, var, index, t);
    }

    void ActionSelectionSawtoothLP::createOmegaConstraints(IloEnv &env, IloRangeArray &con, IloNumVarArray &var, number &index, number t)
    {
        number recover = 0;
        // Go over all points in the point set at t+1
        for (const auto &point_k : getSawtoothValueFunction()->getRepresentation(t + 1))
        {
            const auto &next_occupancy_state = point_k.first->toOccupancyState();

            // Build constraint \sum{x',o'} \omega_k(x',o') = 1
            con.add(IloRange(env, 1.0, 1.0));

            // Go over all Joint History Next
            for (const auto &next_jhistory : next_occupancy_state->getJointHistories())
            {
                // <! \omega_k(x',o')
                auto VarName = this->getVarNameWeightedStateJointHistory(next_occupancy_state, next_jhistory);
                recover = this->getNumber(VarName);
                con[index].setLinearCoef(var[recover], +1.0);
            }
            index++;
        }
    }

    void ActionSelectionSawtoothLP::createInitialConstraints(const std::shared_ptr<OccupancyStateInterface> &occupancy_state, IloEnv &env, IloRangeArray &con, IloNumVarArray &var, number &index, number t)
    {
        auto underlying_problem = ActionSelectionBase::getWorld()->getUnderlyingProblem();
        auto relaxation = std::static_pointer_cast<RelaxedValueFunction>(getSawtoothValueFunction()->getInitFunction());
        number recover = 0;
        double coef;

        con.add(IloRange(env, -IloInfinity, 0));
        con[index].setLinearCoef(var[this->getNumber(this->getVarNameWeight(0))], +1.0);

        // Go over all actions
        for (const auto &u : *underlying_problem->getActionSpace(t))
        {
            auto action = u->toAction();
            for (const auto &joint_history : occupancy_state->getJointHistories())
            {
                // Compute coefficient related to a(u|o)
                coef = occupancy_state->getProbability(joint_history) * relaxation->getQValueAt(occupancy_state->getBeliefAt(joint_history), action, t);
                //<! 1.c.4 get variable a(u|o) and set constant
                recover = this->getNumber(this->getVarNameJointHistoryDecisionRule(action, joint_history));

                con[index].setLinearCoef(var[recover], -coef);
            }
        }
        index++;
    }

    void ActionSelectionSawtoothLP::createSawtoothBigM(const std::shared_ptr<OccupancyStateInterface> &occupancy_state, const std::shared_ptr<OccupancyStateInterface> &next_occupancy_state, const std::shared_ptr<JointHistoryInterface> &next_joint_history, const std::shared_ptr<Observation> &next_observation, IloEnv &env, IloRangeArray &con, IloNumVarArray &var, number &index, number t)
    {
        auto underlying_problem = ActionSelectionBase::getWorld()->getUnderlyingProblem();
        number recover = 0;
        double coef;

        con.add(IloRange(env, -IloInfinity, this->bigM_value_));
        con[index].setLinearCoef(var[this->getNumber(this->getVarNameWeight(0))], +1.0);

        // Go over all actions
        for (const auto &u : *underlying_problem->getActionSpace(t))
        {
            auto action = u->toAction();
            for (const auto &joint_history : occupancy_state->getJointHistories())
            {

                // Compute coefficient related to a(u|o)
                coef = getSawtoothValueFunction()->getSawtoothValueAt(/* support from current point */ occupancy_state, joint_history, action, /* witness point */ next_occupancy_state, /* support of witness point */ next_joint_history, /* to be kept */ next_observation, t, false);
                // std::cout << occupancy_state->getProbability(joint_history) << " = " << coef << std::endl;

                //<! 1.c.4 get variable a(u|o) and set constant
                recover = this->getNumber(this->getVarNameJointHistoryDecisionRule(action, joint_history));

                con[index].setLinearCoef(var[recover], -coef);
            }
        }
        // <! \omega_k(x',o') * BigM
        recover = this->getNumber(this->getVarNameWeightedStateJointHistory(next_occupancy_state, next_joint_history));
        con[index].setLinearCoef(var[recover], this->bigM_value_);

        index++;
    }

    void ActionSelectionSawtoothLP::createSawtoothIloIfThen(const std::shared_ptr<OccupancyStateInterface> &occupancy_state, const std::shared_ptr<OccupancyStateInterface> &next_occupancy_state, const std::shared_ptr<JointHistoryInterface> &next_joint_history, const std::shared_ptr<Observation> &next_observation, IloEnv &env, IloModel &model, IloNumVarArray &var, number t)
    {
        auto underlying_problem = ActionSelectionBase::getWorld()->getUnderlyingProblem();
        number recover = 0;

        IloExpr expr(env);
        //<! 1.c.1 get variable v and set coefficient of variable v
        expr = var[this->getNumber(this->getVarNameWeight(0))];

        // Go over all actions
        for (const auto &u : *underlying_problem->getActionSpace(t))
        {
            auto action = u->toAction();
            for (const auto &joint_history : occupancy_state->getJointHistories())
            {
                //<! 1.c.4 get variable a(u|o) and set constant
                recover = this->getNumber(this->getVarNameJointHistoryDecisionRule(action, joint_history));

                // std::cout << "#> p(" << joint_history->short_str() << ")=" << occupancy_state->getProbability(joint_history) << std::endl;
                double coef = getSawtoothValueFunction()->getSawtoothValueAt(/* support from current point */ occupancy_state, joint_history, action, /* witness point */ next_occupancy_state, /* support of witness point */ next_joint_history, /* to be kept */ next_observation, t, false);
                expr -= var[recover] * coef;
            }
        }

        // <! get variable \omega_k(x',o')
        recover = this->getNumber(this->getVarNameWeightedStateJointHistory(next_occupancy_state, next_joint_history));
        model.add(IloIfThen(env, var[recover] > 0, expr <= 0));
    }

    std::shared_ptr<Joint<std::shared_ptr<Observation>>> ActionSelectionSawtoothLP::determineNextJointObservation(const std::shared_ptr<JointHistoryInterface> &next_joint_history, number)
    {
        return std::static_pointer_cast<Joint<std::shared_ptr<Observation>>>(next_joint_history->getLastObservation());
    }

    std::shared_ptr<SawtoothValueFunction> ActionSelectionSawtoothLP::getSawtoothValueFunction() const
    {
        return this->sawtooth_vf;
    }

    std::shared_ptr<Action> ActionSelectionSawtoothLP::getVariableResult(const std::shared_ptr<ValueFunctionInterface> &vf, const std::shared_ptr<State> &state, const IloCplex &cplex, const IloNumVarArray &var, number t)
    {
        number recover = 0;
        // Go over all points in the point set at t+1
        for (const auto &point_k : this->getSawtoothValueFunction()->getRepresentation(t + 1))
        {
            const auto &next_occupancy_state = point_k.first->toOccupancyState();

            // Go over all Joint History Next
            for (const auto &next_jhistory : next_occupancy_state->getJointHistories())
            {
                // <! \omega_k(x',o')
                auto VarName = this->getVarNameWeightedStateJointHistory(next_occupancy_state, next_jhistory);
                recover = this->getNumber(VarName);

                // Add the variable in the vector only if the variable is true
            }
        }

        // Create the JointDeterminiticDecisionRule
        return DecentralizedLP::getVariableResult(vf, state, cplex, var, t);
    }
}

#endif