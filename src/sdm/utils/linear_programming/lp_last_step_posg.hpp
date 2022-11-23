#pragma once
#include <sdm/utils/linear_programming/variable_naming.hpp>
#include <sdm/utils/value_function/value_function.hpp>
#include <sdm/world/occupancy_mg.hpp>
#include <sdm/core/state/occupancy_state.hpp>
#include <sdm/core/state/private_occupancy_state.hpp>
#include "gurobi_c++.h"

namespace sdm
{
    class LPLastStepPOSG : public VarNaming
    {
    public:
        LPLastStepPOSG();
        LPLastStepPOSG(const std::shared_ptr<SolvableByHSVI> & world, number agent_id_);
        ~LPLastStepPOSG();

        /**
         * @brief Get the world 
         */
        std::shared_ptr<OccupancyMG> getWorld() const;

        /**
         * @brief Main function who is used to create the Linear program and solve it.
         * 
         * @param occupancy_state the occupancy state
         * @param t the time step
         * @return the decision rule 
         */
        Pair<std::shared_ptr<StochasticDecisionRule>, std::map<std::shared_ptr<HistoryInterface>,double>> createLP(const std::shared_ptr<OccupancyStateMG> &occupancy_state, number t);

        /**
         * @brief Create the variable which will be used to resolve the LP
         * 
         * @param occupancy_state 
         * @param env 
         * @param var 
         * @param t 
         */
        void createVariables(const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBEnv &env, GRBVar* varVariables,number &index, number t, number agent_id,GRBModel & model);
        
        /**
         * @brief Create the constraints of the LP
         * 
         * @param occupancy_state 
         * @param env 
         * @param con 
         * @param var 
         * @param index 
         * @param t 
         */
        void createConstraints(const std::shared_ptr<OccupancyStateMG>& occupancy_state, GRBEnv &env, GRBModel &model, GRBConstr* varContraints, GRBVar* varVariables, number &index, number t, number agent_id);
        

        GRBLinExpr createObjectiveFunction(const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBVar* varVariables, number t);


        /**
         * @brief Get the result of the variable created
         * 
         * @param occupancy_state 
         * @param cplex 
         * @param var 
         * @param t 
         * @return std::shared_ptr<Action> 
         */
        std::shared_ptr<StochasticDecisionRule> getVariableResult(const std::shared_ptr<OccupancyStateMG> &occupancy_state,const GRBModel &gurobiModel, const GRBVar* varVariablesVarArray, number t, number agent_id);

        std::shared_ptr<JointHistoryInterface> getRevertedHistory(std::shared_ptr<HistoryInterface> h1, std::shared_ptr<HistoryInterface> h2, const std::shared_ptr<OccupancyStateMG>& occupancy_state);

    std::shared_ptr<sdm::JointAction> getActionPointer(std::shared_ptr<sdm::Action> a, std::shared_ptr<sdm::Action> b,
    std::shared_ptr<POMDPInterface> pomdp, number agent_id_) const;

    std::map<std::shared_ptr<HistoryInterface>, double> historyValues;
    protected:
        /**
         * @brief The world
         */
        std::shared_ptr<OccupancyMG> world_;
        number agent_id_;
        number opponent_id_;
        int nbActions=0;
        int nbHistories=0;
        int nbHistoriesOpponent=0;
        int nbActionsOpponent=0;
    };
}
