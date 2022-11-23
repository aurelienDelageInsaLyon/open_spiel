#include <sdm/utils/value_function/action_selection/lp/action_maxplan_lp_mg.hpp>
#include <sdm/utils/linear_programming/lp_problem.hpp>
#include <sdm/world/base/mpomdp_interface.hpp>
#include <sdm/core/action/stochastic_dr_posg.hpp>
#include <sdm/core/state/jhistory_tree.hpp>

namespace sdm
{
    ActionSelectionMaxplanLPMG::ActionSelectionMaxplanLPMG() {}

    ActionSelectionMaxplanLPMG::ActionSelectionMaxplanLPMG(const std::shared_ptr<SolvableByHSVI> &world, number agent_id_) : world_(std::dynamic_pointer_cast<OccupancyMG>(world)), agent_id_(agent_id_) {}

    ActionSelectionMaxplanLPMG::~ActionSelectionMaxplanLPMG() {}

    std::shared_ptr<OccupancyMG> ActionSelectionMaxplanLPMG::getWorld() const
    {
        return this->world_;
    }

    Pair<std::shared_ptr<StochasticDecisionRule>, std::map<std::shared_ptr<HistoryInterface>, double>> ActionSelectionMaxplanLPMG::createLP(std::shared_ptr<partialQValueFunction> &vf, const std::shared_ptr<OccupancyStateMG> &state, number t)
    {   
        this->gameMatrix = *(new std::map<std::pair<std::pair<std::shared_ptr<HistoryInterface>,std::shared_ptr<Action>>, std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>>,double>);
        this->gameMatrixFromVoid = *(new std::map<std::pair<std::shared_ptr<HistoryInterface>,std::shared_ptr<Action>>,double>);

        this->dualValues = *(new std::map<std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>, double>);

        //std::cout << "\n lp timestep 0, agent_id_ : " << this->agent_id_;
        //std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n";
        this->timestep = t;
        this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)->str();

        this->nbActions = this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)->toDiscreteSpace()->getNumItems();

        this->nbActionsOpponent = this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((this->agent_id_+1)%2)->toDiscreteSpace()->getNumItems();

        this->nbHistories = state->getIndividualHistories(this->agent_id_).size();
        number index = 0;
        this->nbHistoriesOpponent = state->getIndividualHistories((this->agent_id_+1)%2).size();
        // clear the set of variable names
        this->variables.clear();

        //<! greedy decision, initialization
        //std::shared_ptr<Action> action;
        std::shared_ptr<StochasticDecisionRule> strategy = std::make_shared<StochasticDecisionRule>(this->nbActions);

        std::vector<double> value;

        GRBEnv env;
        //env.set("LogFile", "logGurobiLastStep.log");
        env.start();
        std::map<std::shared_ptr<HistoryInterface>, double> listValues;

        try
        {
            //std::cout << "\n nb actions : " << this->nbActions << " nb histories : " << this->nbHistories << " nb histories opponent : " << this->nbHistoriesOpponent << " nb actions opponent : " << this->nbActionsOpponent;
            GRBModel model = GRBModel(env);
            int nbConstraints = this->nbHistories + vf->getListWTuples(t+1).size();
            int nbVariables = this->nbActions*this->nbHistories + 1;
            //std::cout << "\n nb constraints : " << nbConstraints;
            //std::cout << "\n nbVariables : " << nbVariables;
            GRBConstr constraints[nbConstraints];
            GRBVar variables[nbVariables];

            //std::cout << "\n nb variables : " << nbVariables << "\n nb constraints : " << nbConstraints;

            //std::cout << "\ncreating the variables\n";
            this->createVariables( state, env, variables, index, t, this->agent_id_, model);

            //std::cout << "\nsetting the objective\n";
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

            //std::cout << "\ncreating the constraints\n"<<std::flush;
            this->createConstraints(vf, state, env, model, constraints, variables, index, t, this->agent_id_);
            //std::cout << "\n end creating constraints"<<std::flush;
            
            if (this->agent_id_==0){
            model.write("../lpMax.lp");
            }
            if (this->agent_id_==1){
            model.write("../lpMin.lp");
            }
            
            ///////  END CORE  CPLEX Code ///////

            //std::cout << "trying to optimize";
            model.optimize();
            //value = model.get(variables[nbVariables]);

            //std::cout << "\n end optimizing, trying to retrieve values";
            //std::cout << "\n trying to get strategies"<<std::flush;
            //extract the strategies
            //-----------------------------------
            int i = 0;
            for (auto & hist : state->getIndividualHistories(this->agent_id_)){
                int j=0;
                for (auto & action : *this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)){
                    strategy->setProbability(hist,action->toAction(),variables[i*this->nbActions+j].get(GRB_DoubleAttr_X));
                    j++;
                }
                i++;
            }
            
            //std::cout << "\n end trying to get strategies"<<std::flush;
            //-----------------------------------

            //std::cout << "\n getting the dual values" << std::flush;
            //get the dual values
            //-----------------------------------
            
            std::map<std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>, double> resDualValues;
            int index = 0;
            GRBConstr* constrForDual =  model.getConstrs();
            for (const auto & wTuple : vf->getListWTuples(t+1))
            {
                double probaW = constrForDual[this->nbHistories+ index].get(GRB_DoubleAttr_Pi);
                //std::cout << " \n dual value : " << probaW;
                resDualValues.emplace(wTuple,probaW);
                index++;
            }
            this->dualValues = resDualValues;

            //-----------------------------------

            std::map<std::shared_ptr<HistoryInterface>,double> res;
            double finalValue = 0.0;
            for (const auto &history : state->getIndividualHistories(this->agent_id_))
            {                   
                int indexAction = 0;
                    //auto  joint_history_reverted = this->getRevertedHistory(history,indiv_history_opponent,occupancy_state);
                double valMax = ((this->agent_id_*2)-1)*INFINITY;

                // agent 0, maximizing -> valMax = -infini, Ok for maximizing
                if (state->getProbabilityOverIndividualHistories(this->agent_id_,history)>0.0){
                    for (const auto& action : *(this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)))
                    {
                        double valAction = 0.0;
                        
                        if (vf->getListWTuples(t+1).size()>0){
                        for (const auto & wTuple : vf->getListWTuples(t+1))
                        {  
                            valAction += dualValues[wTuple]*gameMatrix[std::make_pair(std::make_pair(history,action->toAction()),wTuple)];
                            //std::cout << "\n val game matrix : " << gameMatrix[std::make_pair(std::make_pair(history,action->toAction()),wTuple)];
                        }
                        if (valAction>valMax && this->agent_id_==0){
                            valMax = valAction;
                        }
                        if (valAction<valMax && this->agent_id_==1){
                            valMax=valAction;
                        }
                        }
                        else{
                            valAction=gameMatrixFromVoid[std::make_pair(history,action->toAction())];

                        if (valAction>valMax && this->agent_id_==0){
                            valMax = valAction;
                        }
                        if (valAction<valMax && this->agent_id_==1){
                            valMax=valAction;
                        }
                        }
                    }
                    //std::cout << "\n valmax : " << (valMax*state->getProbabilityOverIndividualHistories(this->agent_id_,history));
                    finalValue+=valMax;
                    res.emplace(history,valMax/state->getProbabilityOverIndividualHistories(this->agent_id_,history));
                }
            }

            //std::cout << "\n returning value lp timestep : " << this->timestep << " : " << finalValue << " , " << " and opt : " << variables[this->nbHistories*this->nbActions].get(GRB_DoubleAttr_X)<<std::flush ;
            //std::exit(1);
            /*
            std::cout << "\n strategy returned : " << strategy->str();
            if (t==1 && (finalValue !=  variables[this->nbHistories*this->nbActions].get(GRB_DoubleAttr_X))){
                std::exit(1);
            }*/
            return std::make_pair(strategy,res);
        }
        catch (GRBException &e)
        {
            std::cout << "\n timestep : " << t <<std::flush;
            std::cerr << "intermediate timestep, Concert exception caught: " << e.getMessage() << std::endl;
            std::exit(1);
        }
        catch (const std::exception &exc)   
        {
            std::cerr << "Non-Concert exception caught: " << exc.what() << std::endl;
            std::exit(1);
        }
        
        return std::make_pair(strategy, listValues);
    }


    void ActionSelectionMaxplanLPMG::createVariables(const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBEnv &env, GRBVar* varVariables,number &index, number t, number agent_id, GRBModel & model)
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
                varVariables[j*this->nbActions + i] = model.addVar(0.0,1.0,0.0,GRB_CONTINUOUS,std::to_string((j*this->nbActions+i)));
                //this->setNumber(VarName, index++);
                i++;
            }
            j++;
        }

       // int k = 0;
        //for (const auto& indiv_history : occupancy_state->getIndividualHistories((agent_id+1)%2))
        //{
        varVariables[this->nbActions*this->nbHistories] = model.addVar(-std::numeric_limits<double>::infinity(),std::numeric_limits<float>::infinity(),1.0,GRB_CONTINUOUS,std::to_string(this->nbActions*this->nbHistories));
            //k++;
        //}
    }

    void ActionSelectionMaxplanLPMG::createConstraints(std::shared_ptr<partialQValueFunction> &vf, const std::shared_ptr<OccupancyStateMG>& occupancy_state, GRBEnv &env, GRBModel &model, GRBConstr* varConstraints, GRBVar* varVariables, number &index, number t, number agent_id)
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
        
        int indexW = 0;

        if (vf->getListWTuples(t+1).size()>0){
        //for now, this is unfortunately void
        for (const auto & wTuple : vf->getListWTuples(t+1))
        {
            GRBLinExpr qexpr =0;
            int indexHistory = 0;

            for (const auto &history : occupancy_state->getIndividualHistories(this->agent_id_))
            {                   
                int indexAction = 0;

                for (const auto& action : *(this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)))
                {

                    //std::cout << "\n occ : " << occupancy_state->str();   
                    double reward = this->computeReward(occupancy_state,history,action->toAction(),wTuple);

                    //std::cout << "\n computed reward : " << reward << std::flush;
                    //std::shared_ptr<Action> jact = this->getActionPointer(action->toAction(),action_opponent->toAction(),this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),this->agent_id_);
                        
                    double valueNextStep = this->computeValueNextStep(occupancy_state,wTuple, history,action->toAction());

                    //double valLipschitz = this->getLispchitzValue(wTuple, occupancy_state,history,action);

                    qexpr+= (-1.0)*varVariables[indexHistory*this->nbActions+indexAction] * (reward+valueNextStep);

                    //std::cout << "\n reward" << reward << " + valuenextstep : " << valueNextStep<< " agent id :"  << this->agent_id_;
                    //std::cout << "\n putting in game matrix : " <<reward+valueNextStep;
                    double valToPut = reward+valueNextStep;
                    //std::cout << "\n map already contains? " << gameMatrix.count(std::make_pair(std::make_pair(history,action->toAction()),wTuple));
                    gameMatrix.emplace(std::make_pair(std::make_pair(history,action->toAction()),wTuple),valToPut);
                    //std::cout << "\n putted in game matrix : " <<gameMatrix[std::make_pair(std::make_pair(history,action->toAction()),wTuple)];

                    indexAction++;
                }
                indexHistory++;
            }
            qexpr+= varVariables[this->nbHistories*this->nbActions]*1.0;//*((this->agent_id_*2)-1);
                        

            //std::cout << "\n qexpr : " << qexpr;
            indexW++;
                
            //TODO : check the "<"
            if (this->agent_id_==0){
            varConstraints[this->nbHistories] = model.addConstr(qexpr,'<',0.0);
            }
            else{
                varConstraints[this->nbHistories] = model.addConstr(qexpr,'>',0.0);
            }
            //}
            indexW++;
        }
        }
        else{
            //int indexW =0;

            //for (const auto& action_opponent : *( this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((this->agent_id_+1)%2)))
            //{
            GRBLinExpr qexpr =0;
            int indexHistory = 0;

            for (const auto &history : occupancy_state->getIndividualHistories(this->agent_id_))
            {                   
                int indexAction = 0;
                    
                    //auto  joint_history_reverted = this->getRevertedHistory(history,indiv_history_opponent,occupancy_state);

                for (const auto& action : *(this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(this->agent_id_)))
                {
                    //double reward = this->computeReward(occupancy_state,history,action->toAction(),wTuple);
                    double reward = this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getMaxReward();
                    //std::cout << "\n computed reward : " << reward;
                    //std::shared_ptr<Action> jact = this->getActionPointer(action->toAction(),action_opponent->toAction(),this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),this->agent_id_);
                    double valueNextStep = this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getMaxReward()*(this->world_->getHorizon()-t);
                    if (this->agent_id_==1){
                    valueNextStep = this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getMinReward()*(this->world_->getHorizon()-t);
                    }

                    qexpr+= (-1.0)*varVariables[indexHistory*this->nbActions+indexAction] * (reward+valueNextStep);

                    gameMatrixFromVoid.emplace(std::make_pair(history,action->toAction()),reward+valueNextStep);

                    indexAction++;
                }
                indexHistory++;
            }
            qexpr+= varVariables[this->nbHistories*this->nbActions]*1.0;//*((this->agent_id_*2)-1);
            //indexW++;
            //TODO : check the "<"
            if (this->agent_id_==0){
            varConstraints[this->nbHistories] = model.addConstr(qexpr,'<',0.0);
            }
            else{
                varConstraints[this->nbHistories] = model.addConstr(qexpr,'>',0.0);
            }
            //}
            //indexW++;
        }
    }

    double ActionSelectionMaxplanLPMG::getLispchitzValue(const std::tuple<std::shared_ptr<sdm::OccupancyStateMG>, std::shared_ptr<sdm::StochasticDecisionRule>, std::shared_ptr<sdm::VectorMG>> & wTuple, const std::shared_ptr<OccupancyStateMG> & state, const std::shared_ptr<HistoryInterface> & history, const std::shared_ptr<Action> & action,const  std::shared_ptr<Observation> & observation) const {
        

        if (state->getProbabilityOverIndividualHistories(this->agent_id_,history)>0.0){
        /*
        std::cout << "\n timestep : " << this->timestep;
        std::cout << "\n computing lipschitz component";
        std::cout << "\n dr : " << std::get<1>(wTuple)->str();
        std::cout << "\n next occ : " << std::get<2>(wTuple)->occ_support->str();
        std::cout << " \n occ : " << std::get<0>(wTuple)->str();*/
        //std::shared_ptr<PrivateBrOccupancyState> sigmaC1 = std::make_shared<PrivateBrOccupancyState>(this->agent_id_,2,this->world_->getHorizon()-t-1,std::get<2>(wTuple));
        
        std::shared_ptr<PrivateBrOccupancyState> sigmaC1 = std::make_shared<PrivateBrOccupancyState>(*state->getPrivateOccupancyState(this->agent_id_,history), this->agent_id_, *std::get<1>(wTuple));
        //sigmaC1->initialState_=state;

        std::shared_ptr<HistoryInterface> nextHistory = history->expand(observation,action);

        double norm1 = 0.0;
        
        std::shared_ptr<PrivateBrOccupancyState> nextSigmaC1 = std::dynamic_pointer_cast<PrivateBrOccupancyState>(sigmaC1->computeNext(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),action,observation,this->timestep).first);

        /*
        std::cout << "\n done transiting for base cond distrib";
        std::cout << "\n hist : " << history->str();
        std::cout << "\n last state : " << std::get<0>(wTuple)->str();
        */

        if (std::get<0>(wTuple)->getProbabilityOverIndividualHistories(this->agent_id_,history)>0.0){
            
            //std::cout << "\n trying to get cond distrib"<<std::flush;
            std::shared_ptr<PrivateBrOccupancyState> sigmaC1Tuple = std::make_shared<PrivateBrOccupancyState>(*std::get<0>(wTuple)->getPrivateOccupancyState(this->agent_id_,history), this->agent_id_, *std::get<1>(wTuple));
            //std::cout << "\n done getting cond distrib"<<std::flush;

            //std::cout << "\n trying to transitate cond distrib"<<std::flush;
            std::shared_ptr<PrivateBrOccupancyState> nextSigmaC1Tuple = std::dynamic_pointer_cast<PrivateBrOccupancyState>(sigmaC1Tuple->computeNext(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),action,observation,this->timestep).first);
            //std::cout << "\n done trying to transitate cond distrib"<<std::flush;



            //std::cout << "\n next history : " << nextHistory->short_str();


            norm1+= nextSigmaC1->norm_1_other(nextSigmaC1Tuple);


        }
        else{
            //std::cout << "\n returning norm1 because other is null :  " << nextSigmaC1->norm_1();
            //std::exit(1);
            return nextSigmaC1->norm_1();
        }
        //norm1+= nextSigmaC1->norm_1_other(std::get<2>(wTuple)->getSigmaC(nextHistory));
        //std::cout << "\n timestep : " << this->timestep << "returning norm 1 : " << norm1;

        
        //std::exit(1);
        //std::cout << "\n returning norm 1 : " << norm1 << " between : " << state->str() << " and " << std::get<0>(wTuple)->str();
        
        return norm1;
        }

        return 0.0;
    }
    /*
    auto pomdp = std::dynamic_pointer_cast<MPOMDP>(mdp);

        //auto map = std::dynamic_pointer_cast<TabularStateDynamics>(pomdp->getStateDynamics());

        auto listActions = 
            std::dynamic_pointer_cast<MultiDiscreteSpace>(pomdp->getActionSpace());
        // For each joint history in the support of the fully uncompressed occupancy state
        auto listActionsCasted = std::dynamic_pointer_cast<DiscreteSpace>
            (listActions->getSpace((this->agent_id_+1)%2));

        this->strategyOther.action_space = *listActionsCasted;
        this->strategyOther.numActions = listActionsCasted->getNumItems();

        //iterating over \neg i's possible histories 
        for (const auto &compressed_joint_history : this->getJointHistories())
        {   

            // Get p(o_t)
            double proba_history = this->getProbability(compressed_joint_history);

            // Get the corresponding belief b_t(s_t,\theta_t^{\neg i})
            auto belief = this->getBeliefAt(compressed_joint_history);

                //iterating over \neg i's possible action for the history compressed_joint_history
                for (const auto &action_other : *listActions->getSpace((this->agent_id_+1)%2)){
                {
                    double proba_action = this->strategyOther.getProbability(compressed_joint_history,action_other->toAction());

                    for (auto jobs : *pomdp->getObservationSpace((this->agent_id_+1)%2,t)){

                    auto jact = this->getActionPointer(action->toAction(),action_other->toAction(),pomdp,this->agent_id_);
                    
                    auto joint_obs = this->getObservationPointer(observation,jobs->toObservation(),pomdp,
                    this->agent_id_);

                    auto [next_belief, proba_observation] = belief->next(mdp, jact->toAction(), joint_obs->toObservation(), t);

                    //proba_observation is the probability to get the joint observation
                        double next_joint_history_probability = proba_history * proba_action * proba_observation;

                        if (next_joint_history_probability > 0)
                        {

                           std::shared_ptr<JointHistoryInterface> next_compressed_joint_history = compressed_joint_history->expand(joint_obs->toObservation(), jact->toAction())->toJointHistory();
                           this->updateOccupancyStateProba(next_one_step_left_compressed_occupancy_state, next_compressed_joint_history, next_belief->toBelief(), next_joint_history_probability);
                        }
                        }
                    }
                }
        }

        //return std::make_pair(next_one_step_left_compressed_occupancy_state,t);
        Pair<std::shared_ptr<State>, double> pair  = this->finalizeNextState(next_one_step_left_compressed_occupancy_state, t);

        auto occ_state = std::dynamic_pointer_cast<sdm::OccupancyState>(pair.first);
        std::shared_ptr<PrivateBrOccupancyState> next_state = std::make_shared<PrivateBrOccupancyState>(*occ_state,this->agent_id_,this->strategyOther);
        
        return std::make_pair(next_state, pair.second);
    */
    Pair<std::shared_ptr<StochasticDecisionRule>, std::map<std::shared_ptr<HistoryInterface>, double>> ActionSelectionMaxplanLPMG::computeGreedyActionAndValue(std::shared_ptr<partialQValueFunction> &vf, const std::shared_ptr<OccupancyStateMG> &occupancy_state, number t){
        return this->createLP(vf,occupancy_state,t);
    }

    std::shared_ptr<JointHistoryInterface> ActionSelectionMaxplanLPMG::getRevertedHistory(std::shared_ptr<HistoryInterface> h1, std::shared_ptr<HistoryInterface> h2, const std::shared_ptr<OccupancyStateMG>& occupancy_state)
    {
        if (this->agent_id_==1){

            for (auto& jh : occupancy_state->getJointHistories()){

                if (jh->getIndividualHistory(1)->short_str() == h1->short_str() && jh->getIndividualHistory(0)->short_str()==h2->short_str()){
                    return jh;
                }
            
            }
        }
        else{
            for (auto& jh : occupancy_state->getJointHistories()){
                
                if (jh->getIndividualHistory(0)->short_str() == h1->short_str() && jh->getIndividualHistory(1)->short_str()==h2->short_str()){
                    return jh;
                }
            }
        }
        std::cout << "\n action_maxplan_lp_mg:: returning null ptr when getting reverted history";
        //std::exit(1);
        return nullptr;
    }

    GRBLinExpr ActionSelectionMaxplanLPMG::createObjectiveFunction(const std::shared_ptr<OccupancyStateMG> &occupancy_state, GRBVar* varVariables, number t)
    {
        GRBLinExpr obj;
        int k = 0;
        /*
        for (const auto& indiv_history : occupancy_state->getIndividualHistories((this->agent_id_+1)%2))
            {
                obj += varVariables[this->nbActions*this->nbHistories+k];
                k++;
            }*/
        
        obj += varVariables[this->nbActions*this->nbHistories];
        return obj;
    }

    std::shared_ptr<StochasticDecisionRule> ActionSelectionMaxplanLPMG::getVariableResult(const std::shared_ptr<OccupancyStateMG> &occupancy_state,const GRBModel &gurobiModel, const GRBVar* varVariablesVarArray, number t, number agent_id)
    {
        number index = 0;

        std::vector<std::shared_ptr<Item>> indiv_histories;
        std::vector<std::pair<std::shared_ptr<Action>,double>> actions;

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
        return std::make_shared<StochasticDecisionRule>(indiv_histories,actions);
    }
    
    //helper functions w.r.t constraints computations
    //----------------------------
    double ActionSelectionMaxplanLPMG::computeReward(const std::shared_ptr<OccupancyStateMG> &occupancy_state, const std::shared_ptr<HistoryInterface> & history, std::shared_ptr<Action> action, const std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>& w)
    {
        double res =0.0;

        //std::cout << "\n trying to compute reward"<<std::flush;
        for (auto & hist_opponent : occupancy_state->getIndividualHistories((this->agent_id_+1)%2)){

            for (auto & action_opponent : *this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((this->agent_id_+1)%2)){

                std::shared_ptr<Action> jact = this->getActionPointer(action->toAction(),action_opponent->toAction(),this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),this->agent_id_);
                auto  joint_history_reverted = this->getRevertedHistory(history,hist_opponent,occupancy_state);
                if (joint_history_reverted!=nullptr){
                    res+=occupancy_state->getProbability(joint_history_reverted)
                        *occupancy_state->getBeliefAt(joint_history_reverted)->getReward(this->world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)
                        *std::get<1>(w)->getProbability(hist_opponent,action_opponent->toAction());
                }
                else{
                    std::cout << "\n history : " << history->short_str() << " , " << hist_opponent->short_str();
                }
            }
        }
        //std::cout << "\n reward : " << res;
        return res;
    }
    
    double ActionSelectionMaxplanLPMG::computeValueNextStep(const std::shared_ptr<OccupancyStateMG> &occupancy_state, const std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>&  tupleW, const std::shared_ptr<HistoryInterface> & history, std::shared_ptr<Action> action)
    {
        double res = 0.0;
        
        auto pomdp = std::dynamic_pointer_cast<POMDPInterface>(getWorld()->getUnderlyingProblem());

        double lambda = (pomdp->getMaxReward() - pomdp->getMinReward())*(this->world_->getHorizon()-this->timestep-1)/2;
        
        std::cout << "\n lambda : " << lambda;
        if (this->agent_id_==1){
            lambda = -lambda;
        }

        //auto map = std::dynamic_pointer_cast<TabularStateDynamics>(pomdp->getStateDynamics());

        auto listActions = 
            std::dynamic_pointer_cast<MultiDiscreteSpace>(pomdp->getActionSpace());
        // For each joint history in the support of the fully uncompressed occupancy state
        auto listActionsCasted = std::dynamic_pointer_cast<DiscreteSpace>
            (listActions->getSpace((this->agent_id_+1)%2));

        //std::cout << "\n listActions : " << listActions->str();
        for (auto & observation : *(std::dynamic_pointer_cast<MPOMDPInterface>(getWorld()->getUnderlyingProblem())->getObservationSpace(this->agent_id_,1))){
            //double valLip = getLispchitzValue(tupleW, occupancy_state, history, action->toAction(), observation->toObservation());
            for (auto & observation_opponent : *(std::dynamic_pointer_cast<MPOMDPInterface>(getWorld()->getUnderlyingProblem())->getObservationSpace((this->agent_id_+1)%2,1))){

                const auto & jobs = this->getObservationPointer(observation->toObservation(),observation_opponent->toObservation(),pomdp, this->agent_id_);

                for (auto & history_opponent : occupancy_state->getIndividualHistories((this->agent_id_+1)%2)){
                    //for (auto & action_opponent : *(std::dynamic_pointer_cast<POMDPInterface>(getWorld()->getUnderlyingProblem())->getA(1)))
                    
                    auto  joint_history_reverted = this->getRevertedHistory(history,history_opponent,occupancy_state);
                    if (joint_history_reverted!=nullptr){
                    for (const auto &action_opponent : *listActions->getSpace((this->agent_id_+1)%2)){
                        
                        std::shared_ptr<Action> jact = this->getActionPointer(action->toAction(),action_opponent->toAction(),this->world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),this->agent_id_);
                        std::shared_ptr<HistoryInterface> nextHist = history->expand(observation->toObservation(),action->toAction());
                        for (const auto & state : *pomdp->getStateSpace(0)){//ok
                        for (const auto & next_state : *pomdp->getStateSpace(0)){//ok
                                res+=std::get<1>(tupleW)->getProbability(history_opponent,action_opponent->toAction())
                                        *occupancy_state->getProbability(joint_history_reverted)*occupancy_state->getBeliefAt(joint_history_reverted)->getProbability(state->toState())
                                        *pomdp->getTransitionProbability(state->toState(),jact,next_state->toState(),0)
                                        *pomdp->getObservationProbability(state->toState(),jact,next_state->toState(),jobs,0)
                                        *(std::get<2>(tupleW)->getValueAt(nextHist));//+lambda*valLip);
                                //std::cout << "\n val next step : " << std::get<2>(tupleW)->getValueAt(nextHist);
                        //std::cout << "\n res : " << res;
                        //std::exit(1);
                               // std::cout << "\n res : " << res;
                            }
                        }

                    }
                    }
                }
            }
        //std::cout << "\n proba marginale : " << occupancy_state->getProbabilityOverIndividualHistories(this->agent_id_,history);
        res += lambda*getLispchitzValue(tupleW, occupancy_state, history, action->toAction(), observation->toObservation())*occupancy_state->getProbabilityOverIndividualHistories(this->agent_id_,history);

        }
//        std::exit(1);

        return res;
    }
    //----------------------------


    //helper functions w.r.t model
    //----------------------------
    std::shared_ptr<sdm::JointAction> ActionSelectionMaxplanLPMG::getActionPointer(std::shared_ptr<sdm::Action> a, std::shared_ptr<sdm::Action> b,
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
    //----------------------------

    std::shared_ptr<sdm::Observation> ActionSelectionMaxplanLPMG::getObservationPointer(std::shared_ptr<sdm::Observation> a, std::shared_ptr<sdm::Observation> b,
    std::shared_ptr<POMDPInterface> pomdp, number agent_id_) const
    {
        //std::cout << "agent_id_ : " << agent_id_;
        if (agent_id_==0){
            for (const auto &joint_observation : *pomdp->getObservationSpace(0)){
                //std::cout << "\n is it a joint obs? " << std::dynamic_pointer_cast<JointItem>(joint_observation);
                
                if (std::dynamic_pointer_cast<JointItem>(joint_observation)->get(0)->str()==a->str()
                && std::dynamic_pointer_cast<JointItem>(joint_observation)->get(1)->str()==b->str()){
                    return joint_observation->toObservation();
                }
            }
        }
        if (agent_id_==1){
            for (const auto &joint_observation : *pomdp->getObservationSpace(0)){
                //std::cout << "\n is it a joint obs? " << std::dynamic_pointer_cast<JointItem>(joint_observation);
                
                if (std::dynamic_pointer_cast<JointItem>(joint_observation)->get(0)->str()==b->str()
                && std::dynamic_pointer_cast<JointItem>(joint_observation)->get(1)->str()==a->str()){
                    return joint_observation->toObservation();
                }
            }
        }
        return nullptr;
    }

    //?
    Pair<std::shared_ptr<Action>, double> ActionSelectionMaxplanLPMG::getGreedyActionAndValue(const std::shared_ptr<ValueFunctionInterface>& vf, const std::shared_ptr<State>& state, number t)
    {

        std::shared_ptr<partialQValueFunction> q_vf = std::dynamic_pointer_cast<partialQValueFunction>(vf);
        std::shared_ptr<OccupancyStateMG> occupancy_state = std::dynamic_pointer_cast<OccupancyStateMG>(state);
        //std::cout << "\n id : " << this->agent_id_;
        auto result = this->createLP(q_vf,occupancy_state,t);        //return result;
        
        double res =-INFINITY;
        
        /*for (auto & p : result.second){
            res+=p.second;
        }*/
        return std::make_pair(result.first,res);
        //std::cout << "\n ActionSelectionMaxplanLPMP shoudl never be called as is, computeGreedyActionAndValue should be called, i'm exiting";
        //std::exit(1);

        //std::shared_ptr<Action> res;
        //return std::make_pair(res,0.0);
    }

    Pair<std::shared_ptr<Action>, std::map<std::shared_ptr<HistoryInterface>,double>> ActionSelectionMaxplanLPMG::getGreedyActionAndValues(const std::shared_ptr<ValueFunctionInterface>& vf, const std::shared_ptr<State>& state, number t)
    {
        std::shared_ptr<partialQValueFunction> q_vf = std::dynamic_pointer_cast<partialQValueFunction>(vf);
        std::shared_ptr<OccupancyStateMG> occupancy_state = std::dynamic_pointer_cast<OccupancyStateMG>(state);
        //std::cout << "\n action maxplan getting optimistic action/value"<<std::flush;
        auto result = this->createLP(q_vf,occupancy_state,t);
        //std::cout << "\n it succed"<<std::flush;
        //std::cout << "\n returned result : " << result.second.begin()->second << " with size : " << result.second.size();
        return result;
        /*
        std::cout << "\n ActionSelectionMaxplanLPMP shoudl never be called as is, computeGreedyActionAndValue should be called, i'm exiting";
        std::exit(1);

        std::shared_ptr<Action> res;
        return std::make_pair(res,0.0);*/
    }
    //

    std::map<std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>, double> ActionSelectionMaxplanLPMG::getTreeDeltaStretegy(){
        return this->dualValues;
    }

}