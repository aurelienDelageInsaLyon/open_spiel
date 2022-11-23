#include <sdm/utils/linear_programming/lp_last_step_posg.hpp>
#include <sdm/utils/linear_programming/lp_problem.hpp>
#include <sdm/world/base/mpomdp_interface.hpp>
#include <sdm/core/action/stochastic_dr_posg.hpp>
#include <sdm/core/state/jhistory_tree.hpp>

namespace sdm
{
    LPLastStepPOSG::LPLastStepPOSG() {}

    LPLastStepPOSG::LPLastStepPOSG(const std::shared_ptr<SolvableByHSVI> &world, number agent_id_) : world_(std::dynamic_pointer_cast<OccupancyMG>(world)), agent_id_(agent_id_) {
        //std::cout << "\n launched with agent_id : " << agent_id_;
        this->opponent_id_ = (agent_id_+1)%2;
        //std::cout << "\n now agent_id : " << agent_id_;
        //std::cout << "\n and opponent_id_ : " << opponent_id_;
        //std::exit(1);
    }

    LPLastStepPOSG::~LPLastStepPOSG() {}

    std::shared_ptr<OccupancyMG> LPLastStepPOSG::getWorld() const
    {
        return this->world_;
    }

    Pair<std::shared_ptr<StochasticDecisionRule>, std::map<std::shared_ptr<HistoryInterface>,double>> LPLastStepPOSG::createLP(const std::shared_ptr<OccupancyStateMG> &state, number t)
    {   
        this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)->str();

        this->nbActions = this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)->toDiscreteSpace()->getNumItems();

        this->nbActionsOpponent = this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->opponent_id_)->toDiscreteSpace()->getNumItems();

        this->nbHistories = state->getIndividualHistories(this->agent_id_).size();
        number index = 0;
        this->nbHistoriesOpponent = state->getIndividualHistories(this->opponent_id_).size();
        // clear the set of variable names
        this->variables.clear();

        //std::cout << "\n nb histories : " << nbHistories;
        //<! greedy decision, initialization
        //std::shared_ptr<Action> action;
        std::shared_ptr<StochasticDecisionRule> strategy = std::make_shared<StochasticDecisionRule>(this->nbActions);

        std::map<std::shared_ptr<HistoryInterface>,double> value;

        GRBEnv env;
        //env.set("LogFile", "logGurobiLastStep.log");
        env.start();
        
        try
        {
            //std::cout << "\n nb actions : " << this->nbActions << " nb histories : " << this->nbHistories << " nb histories opponent : " << this->nbHistoriesOpponent << " nb actions opponent : " << this->nbActionsOpponent;
            GRBModel model = GRBModel(env);
            int nbConstraints = this->nbHistories + this->nbHistoriesOpponent*this->nbActionsOpponent;
            int nbVariables = this->nbActions*this->nbHistories + this->nbHistoriesOpponent;

            GRBConstr constraints[nbConstraints];
            GRBVar variables[nbVariables];

            //std::cout << "\n nb variables : " << nbVariables << "\n nb constraints : " << nbConstraints;

            //std::cout << "\ncreating the variables\n";
            this->createVariables( state, env, variables, index, t, this->agent_id_, model);

            //std::cout << "\nLP last step setting the objective\n";
            GRBLinExpr obj = this->createObjectiveFunction(state,variables,t);
            
            model.setObjective(obj, GRB_MAXIMIZE);

            if (this->agent_id_==0){
            model.setObjective(obj, GRB_MAXIMIZE);
            }
            else{
                model.setObjective(obj, GRB_MINIMIZE);
            }
            index = 0;

            // Create all Constraints of the LP problem

            //std::cout << "\nLP last step creating the constraints\n";
            this->createConstraints(state, env, model, constraints, variables, index, t, this->agent_id_);
            //std::cout << "\n end creating constraints \n";
            model.write("../lpLastStep.lp");

            ///////  END CORE  CPLEX Code ///////

            //std::cout << "trying to optimize";
            model.optimize();
            //value = model.get(variables[nbVariables]);

            


            int i = 0;
            for (auto & hist : state->getIndividualHistories(this->agent_id_)){
                int j=0;
                for (auto & action : *this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)){
                    strategy->setProbability(hist,action->toAction(),variables[i*this->nbActions+j].get(GRB_DoubleAttr_X));
                    j++;
                }
                i++;
            }

            std::map<std::shared_ptr<HistoryInterface>,double> listValues;
            //std::cout << "\n values : ";
            int k = 0;
            double valLastStep = 0.0;
            for (auto& history_opponent : state->getIndividualHistories(this->opponent_id_)){
                if (state->getProbabilityOverIndividualHistories(this->opponent_id_,history_opponent)>0.0){
                 listValues.emplace(history_opponent,variables[this->nbHistories*this->nbActions+k].get(GRB_DoubleAttr_X)/state->getProbabilityOverIndividualHistories(this->opponent_id_,history_opponent));
                }
                if (state->getProbabilityOverIndividualHistories(this->opponent_id_,history_opponent)>0.0){
                //std::cout << " \n value : " << variables[this->nbHistories*this->nbActions+k].get(GRB_DoubleAttr_X)/state->getProbabilityOverIndividualHistories(this->opponent_id_,history_opponent)<< " for player : " << this->agent_id_ << " hist : " << history_opponent->short_str();
                valLastStep += variables[this->nbHistories*this->nbActions+k].get(GRB_DoubleAttr_X);
                            //std::cout << "\n val last step : " << valLastStep;

                }
                k++;
            }
            //std::cout << "\n val last step : " << valLastStep;

            if (agent_id_==0){
		double res = 0.0;
		for (auto & history_opponent : state->getIndividualHistories((agent_id_+1)%2)){
			
			double value=INFINITY;

			for (auto & action_opponent : * world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((agent_id_+1)%2)){

				double val=0.0;

				for (auto & hist : state->getIndividualHistories(agent_id_)){
                    {
					for (auto & action : * world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(agent_id_)){
											
						std::shared_ptr<Action> jact = getActionPointer(action->toAction(),action_opponent->toAction(),world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),agent_id_)->toAction();

						auto  joint_history_reverted = getRevertedHistory(hist,history_opponent,state);
                        if (joint_history_reverted!=0){
						val+= state->getBeliefAt(joint_history_reverted)->getReward(world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)*state->getProbability(joint_history_reverted)*strategy->getProbability(hist,action->toAction());
                        }
					}
                    }
				}
				//std::cout << "\n value : " << val << "for action : " << action_opponent->str();
				if (val<value){
					value=val;
				}

			}
			res+=value;
		}	
		std::cout << " res : " << res;

	}
	else{
		double res = 0.0;
		for (auto & history_opponent : state->getIndividualHistories((agent_id_+1)%2)){
			
			double value=-INFINITY;

			for (auto & action_opponent : * world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((agent_id_+1)%2)){

				double val=0.0;

				for (auto & hist : state->getIndividualHistories(agent_id_)){
					for (auto & action : * world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(agent_id_)){
											
						std::shared_ptr<Action> jact = getActionPointer(action->toAction(),action_opponent->toAction(),world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),agent_id_)->toAction();

						auto  joint_history_reverted = getRevertedHistory(hist,history_opponent,state);
                        if (joint_history_reverted!=0){
						val+= state->getBeliefAt(joint_history_reverted)->getReward(world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)*state->getProbability(joint_history_reverted)*strategy->getProbability(hist,action->toAction());
                        }
					}
				}
				//std::cout << "\n value : " << val << "for action : " << action_opponent->str();
				if (val>value){
					value=val;
				}

			}
			res+=value;
		}
		//std::cout << "\n returning value lp timestep : " << t<< " : " << res;

	}

            //std::exit(1);
            return std::make_pair(strategy,listValues);
            //std::exit(1);
        }
        catch (GRBException &e)
        {
            std::cerr << "lpLastStepPOSG";
            std::cerr << "Concert exception caught: " << e.getMessage() << std::endl;
        }
        catch (const std::exception &exc)
        {
            // catch anything thrown within try block that derives from std::exception
            std::cerr << "Non-Concert exception caught: " << exc.what() << std::endl;
        }

        //return nullptr;
        //env.end();
        
        return std::make_pair(strategy, value);
    }

    void LPLastStepPOSG::createVariables(const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBEnv &env, GRBVar* varVariables,number &index, number t, number agent_id, GRBModel & model)
    {


        int i=0;
        int j=0;        

        //indexed by i
        for (const auto& indiv_history : occupancy_state->getIndividualHistories(agent_id))
        {  
            i=0;
            // Go over all Individual Action
            // indexed by i
            for (const auto& serial_action : * this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_))
            {
                varVariables[j*this->nbActions + i] = model.addVar(0.0,1.0,0.0,GRB_CONTINUOUS, indiv_history->short_str() +std::to_string((j*this->nbActions+i)));
                //this->setNumber(VarName, index++);
                i++;
            }
            j++;
        }

        int k = 0;
        for (const auto& indiv_history : occupancy_state->getIndividualHistories(this->opponent_id_))
        {
            varVariables[this->nbActions*this->nbHistories+k] = model.addVar(-std::numeric_limits<double>::infinity(),std::numeric_limits<float>::infinity(),1.0,GRB_CONTINUOUS,indiv_history->short_str());
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

                constraintProbabilityDistribution += varVariables[j*this->nbActions+i]*1.0;
                i++;
            }
            //<! increment constraints
            varConstraints[j]=model.addConstr(constraintProbabilityDistribution,'=',1.0);
            index++;
            j++;
        }

        //std::cout << "\n constraints for distributions are done\n" << std::flush;
        //now make the constraints for every possible action of the opponent
        int k = 0;
        for (const auto& indiv_history_opponent : occupancy_state->getIndividualHistories(this->opponent_id_))
        {
            int l =0;

            for (const auto& action_opponent : *( this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->opponent_id_)))
            {
                GRBLinExpr qexpr =0;
                int p = 0;

                for (const auto &history : occupancy_state->getIndividualHistories(this->agent_id_))
                {                   
                    int q = 0;
                    
                    auto  joint_history_reverted = this->getRevertedHistory(history,indiv_history_opponent,occupancy_state);
                    if (joint_history_reverted!=0){
                    for (const auto& action : *(this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)))
                    {
                        
                        std::shared_ptr<Action> jact = this->getActionPointer(action->toAction(),action_opponent->toAction(),this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),this->agent_id_);

                        //std::cout << "\n jact : " << occupancy_state->getProbability(joint_history_reverted) << std::flush;

                        qexpr+= (-1.0)*varVariables[p*this->nbActions+q] * occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)*occupancy_state->getProbability(joint_history_reverted);
                        
                        //std::cout << "\n qexpr done " << std::flush;

                        /*
                        std::cout << "\n history : " << joint_history_reverted->short_str() << " action : " << jact->str();
                        occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0);
                        //std::cout << "\n r = " << occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0);
                        std::cout << "\n proba = " <<occupancy_state->getProbability(joint_history_reverted);
                        std::cout << "\n lin coeff : " << occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)*occupancy_state->getProbability(joint_history_reverted);
                        */
                        q++;
                    }
                    }
                    p++;
                }
                qexpr+= varVariables[this->nbHistories*this->nbActions + k]*1.0;//*((this->agent_id_*2)-1);
                l++;
                
                //TODO : check the "<"
                if (this->agent_id_==0){
                varConstraints[this->nbHistories+k*this->nbActionsOpponent+l] = model.addConstr(qexpr,'<',0.0);
                }
                else{
                    varConstraints[this->nbHistories+k*this->nbActionsOpponent+l] = model.addConstr(qexpr,'>',0.0);
                }
            }
            k++;
        }
        //std::cout << "\nfinished";
    }
    //createObjectiveFunction(const std::shared_ptr<ValueFunctionInterface>vf&, const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBVar* varVariables, number t);

    std::shared_ptr<JointHistoryInterface> LPLastStepPOSG::getRevertedHistory(std::shared_ptr<HistoryInterface> h1, std::shared_ptr<HistoryInterface> h2, const std::shared_ptr<OccupancyStateMG>& occupancy_state)
    {
        //std::cout << "\n history opponent : " << h1->short_str() << " history : " << h2->short_str();
        if (this->agent_id_==1){

            for (auto& jh : occupancy_state->getJointHistories()){

                if (jh->getIndividualHistory(1)->short_str() == h1->short_str() && jh->getIndividualHistory(0)->short_str()==h2->short_str()){
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
        //std::cout << "returning null ptr when getting reverted history";
        //std::exit(1);
        return nullptr;
    }

    GRBLinExpr LPLastStepPOSG::createObjectiveFunction(const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBVar* varVariables, number t)
    {
        GRBLinExpr obj;
        int k = 0;
        for (const auto& indiv_history : occupancy_state->getIndividualHistories(this->opponent_id_))
            {
                //recover = this->getNumber(this->getVarNameIndividualHistoryDecisionRule(serial_action->toAction(), indiv_history, agent_id));
                //varConstraints[index].setLinearCoef(varVariables[recover], +1.0);
                obj += varVariables[this->nbActions*this->nbHistories+k];//*occupancy_state->getProbabilityOverIndividualHistories(this->opponent_id_,indiv_history);
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
        if (agent_id_==0){
            for (const auto &joint_action : *pomdp->getActionSpace()){
                if (joint_action->toAction()->toJointAction()->get(0)->str()==a->str()
                && joint_action->toAction()->toJointAction()->get(1)->str()==b->str()){
                    return joint_action->toAction()->toJointAction();
                }
            }
        }
        if (agent_id_==1){
            for (const auto &joint_action : *pomdp->getActionSpace()){
                if (joint_action->toAction()->toJointAction()->get(0)->str()==b->str()
                && joint_action->toAction()->toJointAction()->get(1)->str()==a->str()){
                    return joint_action->toAction()->toJointAction();
                }
            }
        }
        std::cout << "\n i'm returning null ptr when getting actions";
        std::exit(1);
        return nullptr;
    }
}