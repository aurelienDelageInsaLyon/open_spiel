#include <sdm/utils/linear_programming/lp_last_step_posg.hpp>
#include <sdm/utils/linear_programming/lp_problem.hpp>
#include <sdm/world/base/mpomdp_interface.hpp>
#include <sdm/core/action/stochastic_dr_posg.hpp>
#include <sdm/core/state/jhistory_tree.hpp>

namespace sdm
{
    LPLastStepPOSG::LPLastStepPOSG() {}

    LPLastStepPOSG::LPLastStepPOSG(const std::shared_ptr<SolvableByHSVI> &world, number agent_id_) : world_(std::dynamic_pointer_cast<OccupancyMG>(world)), agent_id_(agent_id_) {}

    LPLastStepPOSG::~LPLastStepPOSG() {}

    std::shared_ptr<OccupancyMG> LPLastStepPOSG::getWorld() const
    {
        return this->world_;
    }

    Pair<std::shared_ptr<Action>, double> LPLastStepPOSG::createLP(const std::shared_ptr<OccupancyStateMG> &state, number t)
    {   
        std::cout<< this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)->str()<<std::endl;

        this->nbActions = this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)->toDiscreteSpace()->getNumItems();

        this->nbActionsOpponent = this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((this->agent_id_+1)%2)->toDiscreteSpace()->getNumItems();

        this->nbHistories = state->getIndividualHistories(this->agent_id_).size();
        number index = 0;
        this->nbHistoriesOpponent = state->getIndividualHistories((this->agent_id_+1)%2).size();
        // clear the set of variable names
        this->variables.clear();

        //<! greedy decision, initialization
        std::shared_ptr<Action> action;
        double value;

        GRBEnv env;
        //env.set("LogFile", "logGurobiLastStep.log");
        env.start();
        try
        {
            GRBModel model = GRBModel(env);
            int nbConstraints = this->nbHistories + this->nbHistoriesOpponent*this->nbActionsOpponent;
            int nbVariables = this->nbActions*this->nbHistories + this->nbHistoriesOpponent;

            GRBConstr constraints[nbConstraints];
            GRBVar variables[nbVariables];

            std::cout << "\n nb variables : " << nbVariables << "\n nb constraints : " << nbConstraints;
            // Init the model
            //GRBRangeArray constraints(env);
            //GRBNumVarArray variables(env);
            //model.write("lp.lp");

            /*
            void createVariables(const std::shared_ptr<ValueFunctionInterface>&vf, const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBEnv &env, GRBVar &varVariables,number &index, number t, number agent_id)
            */

            ///////  BEGIN CORE CPLEX Code  ///////

            // Create all Variable of the LP problem
            std::cout << "\ncreating the variables\n";
            this->createVariables( state, env, variables, index, t, this->agent_id_, model);

            // Create the objective function of the LP problem
            //this->createObjectiveFunction(vf, state, variables, obj, t);
            //std::cout << "\n bla : " << variables[this->nbActions*this->nbHistories].index()<<std::flush;

            std::cout << "\nsetting the objective\n";
            GRBLinExpr obj = this->createObjectiveFunction(state,variables,t);
                        model.setObjective(obj, GRB_MAXIMIZE);

            if (this->agent_id_==1){
            model.setObjective(obj, GRB_MAXIMIZE);
            }
            else{
                model.setObjective(obj, GRB_MINIMIZE);
            }
            index = 0;

            // Create all Constraints of the LP problem

            std::cout << "\ncreating the constraints\n";
            this->createConstraints(state, env, model, constraints, variables, index, t, this->agent_id_);

            model.write("../lp2.lp");

            ///////  END CORE  CPLEX Code ///////

            std::cout << "trying to optimize";
            model.optimize();
            //value = model.get(variables[nbVariables]);

            
            for (int i =0;i<this->nbHistories;i++){
                std::cout << "\n history P1 : ";
                for (int j= 0; j<this->nbActions;j++){
                    std::cout << "P("<<j<<"|"<<i<<") = " << variables[i*this->nbActions+j].get(GRB_DoubleAttr_X) << " ";
                }
            }
            std::cout << "\n values : ";
             for (int k=0; k<this->nbHistoriesOpponent;k++){
                std::cout << " value : " << variables[this->nbHistories*this->nbActions+k].get(GRB_DoubleAttr_X);
            }
            
            std::exit(1);
        }
        catch (GRBException &e)
        {
            std::cerr << "Concert exception caught: " << e.getMessage() << std::endl;
        }
        catch (const std::exception &exc)
        {
            // catch anything thrown within try block that derives from std::exception
            std::cerr << "Non-Concert exception caught: " << exc.what() << std::endl;
        }

        //env.end();

        return std::make_pair(action, value);
    }


    void LPLastStepPOSG::createVariables(const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBEnv &env, GRBVar* varVariables,number &index, number t, number agent_id, GRBModel & model)
    {
        //auto underlying_problem = std::dynamic_pointer_cast<OccupancyMG>(this->world_->getUnderlyingProblem());
        //auto occupancy_state = state->toOccupancyState();

        //<! tracking variables

        int i=0;
        int j=0;        

        //int nbActions=this->world_->getBaseActions(agent_id).size();
        //int nbActions = this->world_->getUnderlyingMPOMDP()->getActionSpace(agent_id,t)->toDiscreteSpace()->getNumItems();
        //<! 0.a Build variables  a_i(u_i|o_i)

        //indexed by i
        for (const auto& indiv_history : occupancy_state->getIndividualHistories(agent_id))
        {  
            i=0;
            // Go over all Individual Action
            // indexed by i
            for (const auto& serial_action : * this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_))
            {
                //<! 0.c Build variables a_i(u_i|o_i)
                //VarName = this->getVarNameIndividualHistoryDecisionRule(serial_action->toAction(), indiv_history, agent_id);
                //varVariables.add(GRBBoolVar(env, 0.0, 1.0, VarName.c_str()));
                //varVariables[j*nbActions + i] = model.addVar(0.0,1.0,GRB_CONTINUOUS,VarName.c_str());
                //std::cout << "i would like to put varname : " << static_cast<char>(j*this->nbActions+i)<<std::flush;
                varVariables[j*this->nbActions + i] = model.addVar(0.0,1.0,0.0,GRB_CONTINUOUS,std::to_string((j*this->nbActions+i)));
                //this->setNumber(VarName, index++);
                i++;
            }
            j++;
        }

        int k = 0;
        for (const auto& indiv_history : occupancy_state->getIndividualHistories((agent_id+1)%2))
        {
            varVariables[this->nbActions*this->nbHistories+k] = model.addVar(-std::numeric_limits<double>::infinity(),std::numeric_limits<float>::infinity(),1.0,GRB_CONTINUOUS,std::to_string(this->nbActions*this->nbHistories + k));
            k++;
        }
    }

    void LPLastStepPOSG::createConstraints(const std::shared_ptr<OccupancyStateMG>& occupancy_state, GRBEnv &env, GRBModel &model, GRBConstr* varConstraints, GRBVar* varVariables, number &index, number t, number agent_id)
    {

        number recover = 0;

        int i=0;
        int j=0;

        //indexed by j
        for (const auto& indiv_history : occupancy_state->getIndividualHistories(agent_id))
        {
            i=0;
            //<! 4.a set constraint  \sum_{u_i} a_i(u_i|o_i) = 1
            GRBLinExpr constraintProbabilityDistribution = 0;
            //varConstraints.add(GRBRange(env, 1.0, 1.0));

            //indexed by i
            for (const auto& serial_action : *( this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)))
            {
                //recover = this->getNumber(this->getVarNameIndividualHistoryDecisionRule(serial_action->toAction(), indiv_history, agent_id));
                //varConstraints[index].setLinearCoef(varVariables[recover], +1.0);
                constraintProbabilityDistribution += varVariables[j*this->nbActions+i]*1.0;
                i++;
            }
            //<! increment constraints
            varConstraints[j]=model.addConstr(constraintProbabilityDistribution,'=',1.0);
            index++;
            j++;
        }

        std::cout << "\n constraints for distributions are done\n";
        //now make the constraints for every possible action of the opponent
        int k = 0;
        for (const auto& indiv_history_opponent : occupancy_state->getIndividualHistories((agent_id+1)%2))
        {
            int l =0;

            for (const auto& action_opponent : *( this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((this->agent_id_+1)%2)))
            {
                GRBLinExpr qexpr =0;
                int p = 0;

                for (const auto &history : occupancy_state->getIndividualHistories(this->agent_id_))
                {                   
                    int q = 0;
                    
                    auto  joint_history_reverted = this->getRevertedHistory(indiv_history_opponent,history,occupancy_state);
                    
                    //std::cout << "\n joint_history reverted : " << joint_history_reverted->str();
                    //std::exit(1);
                    
                    for (const auto& action : *(this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)))
                    {
                        
                        std::shared_ptr<Action> jact = this->getActionPointer(action->toAction(),action_opponent->toAction(),this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),this->agent_id_);
                        
                        qexpr+= (-1.0)*varVariables[p*this->nbActions+q] * occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)*occupancy_state->getProbability(joint_history_reverted);
                        
                        /*
                        std::cout << "\n history : " << joint_history_reverted->short_str() << " action : " << jact->str();
                        occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0);
                        //std::cout << "\n r = " << occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0);
                        std::cout << "\n proba = " <<occupancy_state->getProbability(joint_history_reverted);
                        std::cout << "\n lin coeff : " << occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)*occupancy_state->getProbability(joint_history_reverted);
                        */
                        q++;
                    }
                    p++;
                }
                qexpr+= varVariables[this->nbHistories*this->nbActions + k]*1.0;//*((this->agent_id_*2)-1);
                l++;
                //std::cout << "\n adding a constraint, number : " <<this->nbHistories+k*this->nbActionsOpponent+l << "\n";
                //TODO : check the "<"
                if (this->agent_id_==1){
                varConstraints[this->nbHistories+k*this->nbActionsOpponent+l] = model.addConstr(qexpr,'<',0.0);
                }
                else{
                    varConstraints[this->nbHistories+k*this->nbActionsOpponent+l] = model.addConstr(qexpr,'>',0.0);
                }
                //break;
            }
            //break;
            k++;
        }
        std::cout << "\nfinished";
    }
    //createObjectiveFunction(const std::shared_ptr<ValueFunctionInterface>vf&, const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBVar* varVariables, number t);

    std::shared_ptr<JointHistoryInterface> LPLastStepPOSG::getRevertedHistory(std::shared_ptr<HistoryInterface> h1, std::shared_ptr<HistoryInterface> h2, const std::shared_ptr<OccupancyStateMG>& occupancy_state)
    {
        if (this->agent_id_==1){

            for (auto& jh : occupancy_state->getJointHistories()){

                if (jh->getIndividualHistory(0)->short_str() == h2->short_str() && jh->getIndividualHistory(1)->short_str()==h1->short_str()){
                    return jh;
                }
            
            }
        }
        else{
            //std::cout << "\n hist : " << hist->str();
            for (auto& jh : occupancy_state->getJointHistories()){
                
                if (jh->getIndividualHistory(0)->short_str() == h1->short_str() && jh->getIndividualHistory(1)->short_str()==h2->short_str()){
                    return jh;
                }
            }
        }
        std::cout << "returning null ptr";
        return nullptr;
    }

    GRBLinExpr LPLastStepPOSG::createObjectiveFunction(const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBVar* varVariables, number t)
    {
        GRBLinExpr obj;
        int k = 0;
        for (const auto& indiv_history : occupancy_state->getIndividualHistories((this->agent_id_+1)%2))
            {
                //recover = this->getNumber(this->getVarNameIndividualHistoryDecisionRule(serial_action->toAction(), indiv_history, agent_id));
                //varConstraints[index].setLinearCoef(varVariables[recover], +1.0);
                obj += varVariables[this->nbActions*this->nbHistories+k];//*occupancy_state->getProbabilityOverIndividualHistories((this->agent_id_+1)%2,indiv_history);
                k++;
            }
        return obj;
    }

    std::shared_ptr<StochasticDecisionRule> LPLastStepPOSG::getVariableResult(const std::shared_ptr<OccupancyStateMG> &occupancy_state,const GRBModel &gurobiModel, const GRBVar* varVariablesVarArray, number t, number agent_id)
    {
        number index = 0;
        //std::vector<std::shared_ptr<Item>> actions;
        std::vector<std::shared_ptr<Item>> indiv_histories;
        std::vector<std::pair<std::shared_ptr<Action>,double>> actions;
        //auto underlying_problem = std::dynamic_pointer_cast<OccupancyMG>(this->world_->getUnderlyingProblem());
        //auto occupancy_state = state->toOccupancyState();

        // Go over all individual histories
        int i = 0;
        int j = 0;

        for(const auto& ihistory : occupancy_state->getIndividualHistories(agent_id))
        {
            j=0;
            indiv_histories.push_back(ihistory);

            // Go over all individual action
            for(const auto& action : * this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_))
            {

                actions.push_back(std::make_pair(action->toAction(),std::stoi(varVariablesVarArray[j*this->nbActions+i].get(GRB_StringAttr_VarName))));
                i++;
            }
            j++;
        }
        //return std::make_shared<DeterministicDecisionRule>(indiv_histories,actions);
        return std::make_shared<StochasticDecisionRule>(indiv_histories,actions);
    }

    //helper functions
    std::shared_ptr<sdm::JointAction> LPLastStepPOSG::getActionPointer(std::shared_ptr<sdm::Action> a, std::shared_ptr<sdm::Action> b,
    std::shared_ptr<POMDPInterface> pomdp, number agent_id_) const
    {
        if (agent_id_==1){
            for (const auto &joint_action : *pomdp->getActionSpace()){
                if (joint_action->toAction()->toJointAction()->get(0)->str()==a->str()
                && joint_action->toAction()->toJointAction()->get(1)->str()==b->str()){
                    return joint_action->toAction()->toJointAction();
                }
            }
        }
        if (agent_id_==0){
            for (const auto &joint_action : *pomdp->getActionSpace()){
                if (joint_action->toAction()->toJointAction()->get(0)->str()==b->str()
                && joint_action->toAction()->toJointAction()->get(1)->str()==a->str()){
                    return joint_action->toAction()->toJointAction();
                }
            }
        }
        std::cout << "\n i'm returning null ptr";
        return nullptr;
    }
}