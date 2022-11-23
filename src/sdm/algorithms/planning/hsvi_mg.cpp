#include <sdm/types.hpp>
#include <sdm/config.hpp>
#include <sdm/exception.hpp>
#include <sdm/algorithms/planning/hsvi_mg.hpp>
#include <sdm/world/belief_mdp.hpp>
#include <sdm/world/occupancy_mdp.hpp>
#include <sdm/core/state/belief_state.hpp>
#include <sdm/core/state/occupancy_state.hpp>
#include <sdm/core/state/private_occupancy_state.hpp>
#include <sdm/utils/value_function/prunable_structure.hpp>
#include <sdm/utils/value_function/vfunction/sawtooth_value_function.hpp>
#include <sdm/utils/value_function/initializer/pomdp_relaxation.hpp>
#include <sdm/utils/linear_programming/lp_last_step_posg.hpp>
#include <sdm/utils/value_function/update_operator/vupdate/approxW_update.hpp>

#include <cstdlib>
#include <iostream>

#include <sdm/config.hpp>
#include <sdm/exception.hpp>
#include <sdm/parser/parser.hpp>

#include <sdm/world/solvable_by_hsvi.hpp>
#include <sdm/world/occupancy_mdp.hpp>
#include <sdm/algorithms/planning/hsvi_mg.hpp>
#include <sdm/utils/value_function/action_selection/lp/action_maxplan_lp_mg.hpp>
#include <sdm/utils/value_function/update_operator/vupdate/tabular_update.hpp>
#include <sdm/utils/value_function/vfunction/tabular_value_function.hpp>
#include <sdm/utils/value_function/vfunction/partial_Q_value_function.hpp>
#include <sdm/utils/value_function/update_operator/vupdate/approxW_update.hpp>
#include <sdm/world/private_occupancy_mdp.hpp>

namespace sdm
{
    HsviMG::HsviMG(std::shared_ptr<SolvableByHSVI> &world,
               std::shared_ptr<ValueFunction> lower_bound,
               std::shared_ptr<ValueFunction> upper_bound,
               double error,
               number num_max_trials,
               std::string name,
               number lb_update_frequency,
               number ub_update_frequency,
               double time_max,
               bool keep_same_action_forward_backward) : TSVI(world, lower_bound, error, time_max, name),
                                                         lower_bound(lower_bound),
                                                         upper_bound(upper_bound),
                                                         num_max_trials(num_max_trials),
                                                         lb_update_frequency(lb_update_frequency),
                                                         ub_update_frequency(ub_update_frequency),
                                                         keep_same_action_forward_backward(keep_same_action_forward_backward)
    {
        this->lastState = std::make_shared<OccupancyStateMG>(*std::dynamic_pointer_cast<OccupancyStateMG>(world->getInitialState()));
    }

    void HsviMG::initialize()
    {
        TSVI::initialize();
        getUpperBound()->initialize();
    }

    bool HsviMG::stop(const std::shared_ptr<State> &state, double cost_so_far, number t)
    {
        /*if (this->trial==100){
            return true;
        }*/
        if (t==this->getWorld()->getHorizon()){
            return true;
        }
        if (t>0){
            return false;
        }
        return ((excess(state, cost_so_far, t) <= 0.0000001));
    }

    double HsviMG::excess(const std::shared_ptr<State> &state, double cost_so_far, number t)
    {
        double value_excess;
        try
        {
            value_excess = getWorld()->do_excess(getLowerBound()->getValueAt(getWorld()->getInitialState()), getLowerBound()->getValueAt(state, t), getUpperBound()->getValueAt(state, t), cost_so_far, error, t);
        }
        catch (const std::exception &exc)
        {
            // catch anything thrown within try block that derives from std::exception
            std::cerr << "HsviMG::excess(..) exception caught: " << exc.what() << std::endl;
            exit(-1);
        }
        return value_excess;
    }


    void HsviMG::explore(const std::shared_ptr<State> &state, double cost_so_far, number t)
    {   
        std::shared_ptr<StochasticDecisionRule> dr1 = std::make_shared<StochasticDecisionRule>(3);
		std::shared_ptr<StochasticDecisionRule> dr2 = std::make_shared<StochasticDecisionRule>(3);
        this->explore(this->getWorld()->getInitialState(),dr1,dr2,state,cost_so_far,t);
    }


    void HsviMG::explore(const std::shared_ptr<State> &last_state, const std::shared_ptr<StochasticDecisionRule> & lastStrategyP1,const std::shared_ptr<StochasticDecisionRule> & lastStrategyP2, const std::shared_ptr<State> &state, double cost_so_far, number t)
    {       
        //std::cout << OccupancyState::PRECISION;
        //std::cout << Belief::PRECISION;
        //std::exit(1);
        if (t==0 && this->trial>1){
            
            this->testSolution(
            std::dynamic_pointer_cast<partialQValueFunction>(this->getUpperBound())->getSupportVector(1),
            std::dynamic_pointer_cast<partialQValueFunction>(this->getLowerBound())->getSupportVector(0));

        }
            //std::cout << "\n exploring at timestep : " << t;
            if (!stop(state, cost_so_far, t))
            {

                //std::cout << "\n exploring at timestep : " << t;


                std::shared_ptr<Action> actionP1;
                std::shared_ptr<Action> actionP2;
                
                if (t<getWorld()->getHorizon()-1){
                actionP1 = getUpperBound()->getGreedyActionAndValue(state, t).first;
                actionP2 = getLowerBound()->getGreedyActionAndValue(state, t).first;
                }
                else{                
                std::shared_ptr<LPLastStepPOSG> lpP1 = std::make_shared<LPLastStepPOSG>(getWorld(),0);
			    actionP1 = lpP1->createLP(std::dynamic_pointer_cast<OccupancyStateMG>(state),0).first;
                
                std::shared_ptr<LPLastStepPOSG> lpP2 = std::make_shared<LPLastStepPOSG>(getWorld(),1);
			    actionP2 = lpP2->createLP(std::dynamic_pointer_cast<OccupancyStateMG>(state),1).first;

                }

                std::vector<std::shared_ptr<Action>> vec = {actionP1,actionP2};
                std::shared_ptr<JointAction> action = std::make_shared<JointAction>(vec);

                for (const auto &observation : selectObservations(state, action, t))
                {   
                    /*
                    std::shared_ptr<Action> dr1 = std::make_shared<StochasticDecisionRule>(3);
			std::shared_ptr<Action> dr2 = std::make_shared<StochasticDecisionRule>(3);
			//Joint jointDr({dr1,dr2});
			 std::vector<std::shared_ptr<Action>> vec = {dr1,dr2};
            std::shared_ptr<JointAction> jointDr = std::make_shared<JointAction>(vec);
                    */
                    auto next_state = getWorld()->getNextStateAndProba(state,action->toAction(), observation, t).first;
                    //if (t==0 || t==1){
                    //std::cout << "\n strategy : " << action->str();
                    //std::cout << " \n next state : " << next_state->str();
                    //std::exit(1);
                    //}
                    explore(state, std::dynamic_pointer_cast<StochasticDecisionRule>(actionP1),std::dynamic_pointer_cast<StochasticDecisionRule>(actionP2), next_state, cost_so_far + getWorld()->getDiscount(t) * getWorld()->getReward(state, action, t), t + 1);
                }

                if (t<getWorld()->getHorizon()-1){

                    this->updateUpbValue(state, t, lastState, lastStrategyP2);
                    
                    this->updateLobValue(state, t, lastState, lastStrategyP1);
                    
                    
                    //std::cout << "\n value upb t=" <<t<< " : "<< this->getUpperBound()->getValueAt(state,t);
                    //std::cout << "\n value lob t=" <<t<< " : "<< this->getLowerBound()->getValueAt(state,t);
                    
                }
                else{
                    std::shared_ptr<LPLastStepPOSG> lpUpb = std::make_shared<LPLastStepPOSG>(getWorld(),0);
                    auto [dr1, valuesHistP2] = lpUpb->createLP(std::dynamic_pointer_cast<OccupancyStateMG>(state),t);
                    //std::exit(1);

                    std::shared_ptr<LPLastStepPOSG> lpLob = std::make_shared<LPLastStepPOSG>(getWorld(),1);
                    auto [dr2, valuesHistP1] = lpLob->createLP(std::dynamic_pointer_cast<OccupancyStateMG>(state),t);
                    
                    
                    this->updateLastStep(state,last_state,valuesHistP1,valuesHistP2,lastStrategyP1,lastStrategyP2,dr1,dr2,t);

                    std::cout << "\n value upb : " << this->getUpperBound()->getValueAt(state,t);
                    std::cout << "\n value lob : " << this->getLowerBound()->getValueAt(state,t);

                    //std::exit(1);
                    //std::exit(1);
                }
            }

        /*}
        catch (const std::exception &exc)
        {
            // Catch anything thrown within try block that derives from std::exception
            std::cerr << "TSVI::explore(..) exception caught: " << exc.what() << std::endl;
            exit(-1);
        }*/
    }

    // SELECT ACTIONS IN HsviMG
    std::vector<std::shared_ptr<Action>> HsviMG::selectActions(const std::shared_ptr<State> &state, number t)
    {
        return {getUpperBound()->getGreedyAction(state, t)};
    }

    // SELECT OBSERVATION IN HsviMG
    std::vector<std::shared_ptr<Observation>> HsviMG::selectObservations(const std::shared_ptr<State> &state, const std::shared_ptr<Action> &action, number t)
    {
        //std::cout << "action : "<<action->toAction()->str()<< " world : " << this->getWorld()<<std::flush;
        //std::exit(1);
        double error, biggest_error = -std::numeric_limits<double>::max();
        std::shared_ptr<Observation> selected_observation;

        // Select next observation
        auto observation_space = this->getWorld()->getObservationSpaceAt(state, action->toAction(), t);
        //std::cout << "obs space : " << observation_space->str()<< std::flush;
        //std::exit(1);
        double prob_obs = 0.0;
        for (const auto &observation : *observation_space)
        {
            // Get the next state and probability
            auto [next_state, transition_proba] = getWorld()->getNextStateAndProba(state, action->toAction(), observation->toObservation(), t);
            
            //std::cout << "\n next state : " << next_state;
            // Compute error correlated to this next state
            error = transition_proba * excess(next_state, 0, t + 1);
            if (error > biggest_error)
            {
                biggest_error = error;
                selected_observation = observation->toObservation();
                prob_obs = transition_proba;
                
            }
        }
        //std::cout << "\n HsviMG::selected proba obs : " << prob_obs<<std::endl;
        return {selected_observation};
    }

    std::shared_ptr<ValueFunction> HsviMG::getLowerBound() const
    {
        return lower_bound;
    }

    std::shared_ptr<ValueFunction> HsviMG::getUpperBound() const
    {
        return upper_bound;
    }

    void HsviMG::updateLastStep(const std::shared_ptr<State> & state, const std::shared_ptr<State> & last_state,std::map<std::shared_ptr<HistoryInterface>,double> valuesUpb, std::map<std::shared_ptr<HistoryInterface>,double> valuesLob,const std::shared_ptr<StochasticDecisionRule> & lastDr1,const std::shared_ptr<StochasticDecisionRule> & lastDr2,const std::shared_ptr<StochasticDecisionRule> & dr1,const std::shared_ptr<StochasticDecisionRule> & dr2, number t){
        
        std::dynamic_pointer_cast<UpdateW>(this->getUpperBound()->getUpdateOperator())->updateLastStep(state,t,last_state,lastDr2,valuesUpb,dr2);
        std::dynamic_pointer_cast<UpdateW>(this->getLowerBound()->getUpdateOperator())->updateLastStep(state,t,last_state,lastDr1,valuesLob,dr1);

    }

    void HsviMG::updateValue(const std::shared_ptr<State> &state, number t)
    {   
        std::cout << "\n hsvi_mg : should never call this";
        std::exit(1);
        // auto [action, value] = this->getUpperBound()->getGreedyActionAndValue(state, t);
        this->getUpperBound()->getUpdateOperator()->update(state, /* value, */ t);
        this->getLowerBound()->getUpdateOperator()->update(state, /* action, */ t);
    }

    void HsviMG::updateUpbValue(const std::shared_ptr<State> &state, number t, const std::shared_ptr<State> &lastState, const std::shared_ptr<StochasticDecisionRule> & dr)
    {
        std::dynamic_pointer_cast<UpdateW>(this->getUpperBound()->getUpdateOperator())->update(state, /* value, */ t, lastState,dr);
        //std::cout << "\n is it null? " <<  std::dynamic_pointer_cast<UpdateW>(this->getUpperBound()->getUpdateOperator());
    }

    void HsviMG::updateLobValue(const std::shared_ptr<State> &state, number t, const std::shared_ptr<State> &lastState, const std::shared_ptr<StochasticDecisionRule> & dr)
    {
        std::dynamic_pointer_cast<UpdateW>(this->getLowerBound()->getUpdateOperator())->update(state, /* action, */ t, lastState,dr);
    }
    
    //////////////////////////////////////// EXTRACT THE STRATEGY ///////////////////////////////////////////////
    std::shared_ptr<StochasticDecisionRule> HsviMG::extractStrategy(const std::shared_ptr<VectorMG> & vector, int horizon, int agent){
        
        this->rwForTimestep = *(new std::map<int,std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double>>());

        this->recGetRWMix(0,vector, horizon);
        std::map<int,std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double>> mapOfRealizationWeights = this->rwForTimestep;

        /*
        for (auto & bla : mapOfRealizationWeights[0]){
                std:: cout << "\n " << " prefix : " << std::get<0>(bla.first)->short_str() << " suffix : " << std::get<1>(bla.first)->short_str() << " val : " << bla.second;
        }*/ 
        //step 2


        std::map<std::shared_ptr<HistoryInterface>,double> classicalRWs = *(new std::map<std::shared_ptr<HistoryInterface>,double>());

         //we begin by recopying the rws for t=H-1:
        for (auto & bla : mapOfRealizationWeights[0]){
            std::shared_ptr<HistoryInterface> rw = std::get<1>(bla.first);
            classicalRWs.emplace(rw,bla.second);
        }

        std::map<int,std::map<std::shared_ptr<HistoryInterface>,double>> classicalRWsReversed = *(new std::map<int,std::map<std::shared_ptr<HistoryInterface>,double>>());

        /*
        for (auto & bla : classicalRWs){
            std::cout << "\n classic : " << bla.first->str() << " : " << bla.second;
        }
        */

        for (auto & bla : classicalRWs){
            //std::cout << "\n bla : " << bla.first->str() << " : " << bla.second;
            bool alreadyThere = false;
                for (auto & tmp : classicalRWsReversed[horizon-1]){
                    if (std::dynamic_pointer_cast<HistoryTree>(tmp.first)->short_str()==std::dynamic_pointer_cast<HistoryTree>(bla.first)->short_str()){
                        alreadyThere = true;
                    }
                }
            if (alreadyThere==false){
                std::shared_ptr<HistoryTree> tmp = this->revertHistory(std::dynamic_pointer_cast<HistoryTree>(bla.first),nullptr);
                tmp->depth_=horizon-1;
                classicalRWsReversed[horizon-1].emplace(tmp,bla.second);
            }
        }
        
        for (auto & bla : classicalRWsReversed[horizon-1]){
            std::cout << "\n reverted : " << bla.first->str() << " : " << bla.second;
        }


        //step 3 : another try

        for (int t = horizon-1; t>0; t--){
            for (auto & bla : classicalRWsReversed[t]){

                std::shared_ptr<HistoryInterface> histToBeAdded = std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent();
                double valArgMaxBis=0.0;
                double valBis=0.0;
                for (auto & nextHist : classicalRWsReversed[t]){
                    if (histToBeAdded->short_str()==
        std::dynamic_pointer_cast<HistoryTree>(nextHist.first)->getParent()->short_str()){
                        if (nextHist.first->getLastObservation()->str()==sdm::NO_OBSERVATION->str()){
                            valBis+= nextHist.second;
                        }
                    }
                }

                bool alreadyThere = false;
                for (auto & tmp : classicalRWsReversed[t-1]){
                    if (histToBeAdded->short_str()==tmp.first->short_str()){
                        alreadyThere=true;
                    }
                }
                if (alreadyThere==false){
                    classicalRWsReversed[t-1].emplace(histToBeAdded,valBis);
                }
            }
            for (auto & bla : classicalRWsReversed[t-1]){
                std::cout << "\n reverted : " << bla.first->str() << " : " << bla.second;
            }
            //std::exit(1);
            //---------------------------------------------------------------------------
            //now take the argmax
            //
            for (auto & bla : classicalRWsReversed[t-1]){

                //std::shared_ptr<HistoryInterface> histToBeAdded = std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent();
                std::shared_ptr<HistoryTree> histWithNoObs;

                //--------getting the noObs history----------------
                if (bla.first->getPreviousHistory()!=0){
                    histWithNoObs = std::dynamic_pointer_cast<HistoryTree>(bla.first->getPreviousHistory()->expand(
                        sdm::NO_OBSERVATION,std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction()));
                    histWithNoObs->depth_=t-1;
                }
                else{
                    histWithNoObs = std::make_shared<HistoryTree>(std::make_shared<HistoryTree>(), 
                        std::pair(sdm::NO_OBSERVATION,std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction()));
                    histWithNoObs->depth_ = 0;
                }
                //-------------------------------------------------

                //std::cout << "\n hist to be added with no obs : " << histWithNoObs->short_str();
                double argmax = 0.0;
                for (auto & other : classicalRWsReversed[t-1]){
                    if (t>1){
                        if (other.first->getPreviousHistory()->short_str()==bla.first->getPreviousHistory()->short_str()){
                            if (std::dynamic_pointer_cast<HistoryTree>(other.first)->getLastAction()->str()
                                ==std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction()->str()){
                                if (other.second>argmax){
                                    argmax=other.second;
                                }
                            }
                        }
                    }
                    else{
                        if (std::dynamic_pointer_cast<HistoryTree>(other.first)->getLastAction()->str()
                                ==std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction()->str()){
                            if (other.second>argmax){
                                    argmax=other.second;
                            }
                        }
                    }
                }

                bool alreadyThere = false;
                for (auto & tmp : classicalRWsReversed[t-1]){
                    if (histWithNoObs->short_str()==tmp.first->short_str()){
                        alreadyThere=true;
                    }
                }
                if (!alreadyThere){
                    classicalRWsReversed[t-1].emplace(histWithNoObs,argmax);
                }
            }

        }

        for (int t = 0;t<horizon-1;t++){
            for (auto & bla : classicalRWsReversed[t]){
                std::cout << "\n reverted : " << bla.first->str() << " : " << bla.second;
            }
        }
        //std::exit(1);
        //now take the argmax to create (x,(action,-))
        //for (int t = 0; t<horizon-1; t++){
        /*
            for (auto & bla : classicalRWsReversed[t]){

                std::shared_ptr<HistoryInterface> histToBeAdded = std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent();
                std::shared_ptr<HistoryTree> histWithNoObs;

                //--------getting the noObs history----------------
                if (histToBeAdded->getPreviousHistory()!=0){
                    histWithNoObs = std::dynamic_pointer_cast<HistoryTree>(histToBeAdded->getPreviousHistory()->expand(sdm::NO_OBSERVATION,std::dynamic_pointer_cast<HistoryTree>(histToBeAdded)->getLastAction()));
                    histWithNoObs->depth_=t-1;
                }
                else{
                    histWithNoObs = std::make_shared<HistoryTree>(std::make_shared<HistoryTree>(), std::pair(sdm::NO_OBSERVATION,std::dynamic_pointer_cast<HistoryTree>(histToBeAdded)->getLastAction()));
                    histWithNoObs->depth_ = 0;
                }
                //-------------------------------------------------

                double argmax = 0.0;
                for (auto & other : classicalRWsReversed[t]){
                    if (t>0){
                        if (other.first->getPreviousHistory()->str()==bla.first->getPreviousHistory()->str()){
                            if (std::dynamic_pointer_cast<HistoryTree>(other.first)->getLastAction()->str()
                                ==std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction()->str()){
                                if (other.second>argmax){
                                    argmax=other.second;
                                }
                            }
                        }
                    }
                    else{
                        if (std::dynamic_pointer_cast<HistoryTree>(other.first)->getLastAction()->str()
                                ==std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction()->str()){
                            if (other.second>argmax){
                                    argmax=other.second;
                            }
                        }
                    }
                }

                classicalRWsReversed[t].emplace(histWithNoObs,argmax);
            }
        //}
        */


        //step 3 revised with argmax over observations

        //idée : pour le dénominateur, prendre en compte l'observation
        //       pour le numérateur, utiliser les "action,-", qui sont construites enn prenant l'argmax
        //du coup, il faut stocker les (action,observation) et les (action,-)
        


        /*
        for (int t = horizon-1; t>0; t--){
            for (auto & bla : classicalRWsReversed[t]){

                std::shared_ptr<HistoryInterface> histToBeAdded = std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent();

                auto observation_space = this->getWorld()->getObservationSpaceAt(this->getWorld()->getInitialState(),
                    std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction(),t);

                double valArgMax = 0.0;
                double valArgMaxBis = 0.0;

                //creating (x,(action-noObs)) with (x,(action,z),(action,noObs))
                //need to take the argmax over z
                //thus take the argmax over all the possible histToBeAdded !
                for (const auto &observation : *observation_space){
                    double val = 0.0;
                    double valBis = 0.0;

                    for (auto & nextHist : classicalRWsReversed[t]){
                        std::cout << "\n next hist : " << nextHist.first->str() << " trying to create : " << histToBeAdded->str();
                        if (histToBeAdded->short_str()==
        std::dynamic_pointer_cast<HistoryTree>(nextHist.first)->getParent()->short_str()){
                            if (histToBeAdded->getLastObservation()->str()==observation->str()){
                                if (nextHist.first->getLastObservation()->str()==sdm::NO_OBSERVATION->str()){
                                    val+= nextHist.second;
                                }
                            }
                        }
                    }
                    if (val>valArgMax){
                        valArgMax = val;
                    }
                    //here, need to add the rw with (action,observation) to get the denom later
                    //if (val>0.0){
                     //   classicalRWsReversed[t-1].emplace(std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent(),val);
                    //}
                    for (auto & nextHist : classicalRWsReversed[t]){
                        if (histToBeAdded->short_str()==
        std::dynamic_pointer_cast<HistoryTree>(nextHist.first)->getParent()->short_str()){
                            valBis+= nextHist.second;
                        }
                    }

                    if (valBis>valArgMaxBis){
                        valArgMaxBis = val;
                    }
                }

                bool alreadyThere = false;
                bool alreadyThereBis = false;

                for (auto & tmp : classicalRWsReversed[t-1]){
                    if (tmp.first->short_str()==histToBeAdded->short_str()){
                        alreadyThere=true;
                        alreadyThereBis=true;
                    }
                    if (std::dynamic_pointer_cast<HistoryTree>(tmp.first)->getParent()!=0){
                        if(std::dynamic_pointer_cast<HistoryTree>(tmp.first)->getParent()->short_str()
                             == histToBeAdded->getPreviousHistory()->short_str()){
                            if (std::dynamic_pointer_cast<HistoryTree>(tmp.first)->getLastAction()==
                                 std::dynamic_pointer_cast<HistoryTree>(histToBeAdded)->getLastAction()){
                                        alreadyThere=true;
                            }
                        }
                    }
                }
                if (!alreadyThere){
                    // add histToBeAdded with modified last obs.
                    if (histToBeAdded->getPreviousHistory()!=0){
                        //std::dynamic_pointer_cast<HistoryTree>(histToBeAdded)->setLastObservation(sdm::NO_OBSERVATION);
                        std::shared_ptr<HistoryTree> histWithNoObs = std::dynamic_pointer_cast<HistoryTree>(histToBeAdded->getPreviousHistory()->expand(sdm::NO_OBSERVATION,std::dynamic_pointer_cast<HistoryTree>(histToBeAdded)->getLastAction()));
                        histWithNoObs->depth_=t-1;
                        classicalRWsReversed[t-1].emplace(histWithNoObs,valArgMax);
                    }
                    else{
                        std::shared_ptr<HistoryTree> histWithNoObs = std::make_shared<HistoryTree>(std::make_shared<HistoryTree>(), std::pair(sdm::NO_OBSERVATION,std::dynamic_pointer_cast<HistoryTree>(histToBeAdded)->getLastAction()));
                        histWithNoObs->depth_ = 0;
                        classicalRWsReversed[t-1].emplace(histWithNoObs,valArgMax);
                    }
                }
                if (!alreadyThereBis){
                    classicalRWsReversed[t-1].emplace(histToBeAdded,valArgMaxBis);
                }

                // the (action,-) is done, now putting the (action,observation)
                //BUT you also need to put the argmax !!
                /*
                std::string obsDone = "-";
                double val = 0.0;

                std::shared_ptr<HistoryInterface> histToBeAddedBis = std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent();
                std::cout << " \n hist to be added bis : " << histToBeAddedBis->short_str();

                for (auto & nextHist : classicalRWsReversed[t]){


                    if (histToBeAddedBis->short_str()==std::dynamic_pointer_cast<HistoryTree>(nextHist.first)->getParent()->short_str()){
                        if (obsDone != "-"){
                            if (obsDone==nextHist.first->getLastObservation()->str()){
                                val+=nextHist.second;
                                //std::cout << "\n youpii!" << "; value : " << nextHist.second;
                            }
                        }
                        else{
                            val+=nextHist.second;
                            //std::cout << "\n youpii!" << "; value : " << nextHist.second;
                            obsDone=nextHist.first->getLastObservation()->str();
                        }
                    }
                }
                bool alreadyThereBis = false;
                for (auto & tmp : classicalRWsReversed[t-1]){
                    if (tmp.first->short_str()==histToBeAddedBis->short_str()){
                        alreadyThereBis=true;
                    }
                }


                if (!alreadyThereBis){
                    classicalRWsReversed[t-1].emplace(histToBeAddedBis,val);

                }
            }
            //done the (action,-)

    */
        //    }
        //}
        


        //just printing
        for (int t = 0; t<classicalRWsReversed.size();t++){
            for (auto & bla : classicalRWsReversed[t]){
                std::cout << "\n rw reverted : " << bla.first->str() << " : " << bla.second;
            }
        }
        //

       
        std::shared_ptr<StochasticDecisionRule> resStrat = std::make_shared<StochasticDecisionRule>();
        
        //step 4
        for (int t = horizon-1;t>=0; t--){
            for (auto & bla : classicalRWsReversed[t]){
                if (bla.first->getLastObservation()->str()==sdm::NO_OBSERVATION->str()){
                //std::cout << "\n bla.first : " << bla.first->short_str() << " :  " << bla.second;
                if (bla.second > 0.0){
                    double denom = 1.0;
                    if (t>0){
                        for (auto & blabla : classicalRWsReversed[t-1]){
                            /*
                            if (std::dynamic_pointer_cast<HistoryTree>(blabla.first)->short_str()==std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent()->short_str()&&denom==1.0) {
                                denom = blabla.second;
                            }*/
                            if (this->isExpansion(std::dynamic_pointer_cast<HistoryTree>(blabla.first),std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent())){
                                denom = blabla.second;
                            }
                        }
                    }
                    std::shared_ptr<HistoryTree> histToPutInStrat;
                    if (bla.first->getPreviousHistory()==0){
                        histToPutInStrat = std::make_shared<HistoryTree>();
                    }
                    else{
                        histToPutInStrat = std::dynamic_pointer_cast<HistoryTree>(std::dynamic_pointer_cast<HistoryTree>(bla.first)->getParent());
                    }
                    //std::cout << "\n hist : " << histToPutInStrat->short_str()<<std::flush;
                    //std::cout << "\n denom : " << denom;
                    bool found = false;
                    for (auto & elem : resStrat->stratMap){
                        if (elem.first->short_str()==histToPutInStrat->short_str()){
                            //if (resStrat->stratMap[elem.first].count(std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction())==0 
                            //   || resStrat->stratMap[elem.first][std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction()]<=bla.second/denom){
                                resStrat->stratMap[elem.first][std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction()] = bla.second/denom;
                                found = true;
                            //}
                            //else{
                            //    found=true;
                            //}
                        }
                    }
                    if (found==false){
                        resStrat->setProbability(histToPutInStrat,std::dynamic_pointer_cast<HistoryTree>(bla.first)->getLastAction(),bla.second/denom);
                    }
                }
                }
            }
        }
        
        return resStrat;

    }

    bool HsviMG::isExpansion(std::shared_ptr<HistoryTree> h1, std::shared_ptr<HistoryTree> h2){
        //std::cout << "\n isExpansion? " << h1->str() << h2->str()<<std::flush;

        /*
        if (h1->depth_!=h2->depth_){
            std::cout << "\n diff size";
            return false;
        }*/
        if (h1->getLastAction()!=h2->getLastAction() || h1->getLastObservation() != h2->getLastObservation()){
            //std::cout << "\n not an expansion last obs"<<std::flush;
            return false;
        }
        std::shared_ptr<HistoryTree> tmp1 = h1->getParent();
        std::shared_ptr<HistoryTree> tmp2 = h2->getParent();
        
        while (tmp1 != 0){
            if (tmp1!=0 && tmp1->getLastAction()==tmp2->getLastAction() && tmp1->getLastObservation()==tmp2->getLastObservation()){
                tmp1 = tmp1->getParent();
                tmp2 = tmp2->getParent();
            }
            else
            {
                //std::cout << "\n not an expansion"<<std::flush;
                return false;
            }
        }
        //std::cout << "\n indeed is expansion"<<std::flush;
        return true;
    }



    std::shared_ptr<HistoryTree> HsviMG::revertHistory(std::shared_ptr<HistoryTree> head, std::shared_ptr<HistoryTree> tail){
        if (head==nullptr || head->getParent()==nullptr){
            return head;
        }

        std::shared_ptr<HistoryTree> rest = this->revertHistory(head->getParent(),nullptr);

        int a = head->getDepth();
        
        head->depth_= rest->getDepth();

        rest->depth_=a;

        head->getParent()->setParent(head);

        head->setParent(nullptr);

        return rest;
    }

    std::shared_ptr<Action> HsviMG::getRevertLastAction(std::shared_ptr<HistoryTree> h){
        //std::dynamic_pointer_cast<HistoryInterface>(std::dynamic_pointer_cast<HistoryTree>(element.first)->getParent();
        if (h->getDepth()==1){
            return h->getLastAction();
        }
       // std::cout << "\n h : " << h->short_str();
        std::shared_ptr<HistoryTree> res = std::make_shared<HistoryTree>();
        
        std::vector<std::pair<std::shared_ptr<Observation>,std::shared_ptr<Action>>> vectorStore;

        while (h != 0){
            vectorStore.push_back({std::make_pair(h->getLastObservation(),h->getLastAction())});
            h = h->getParent();
        }
        return vectorStore[vectorStore.size()-1].second;
    }

    //////////////////////////////////////// Recursive get cats ///////////////////////////////////////////////

    std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> HsviMG::recGetRWCat(int t, const std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>> & deltaStrat, int horizon){

        std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> rwCat = *(new std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double>());

        //std::cout << "\n recGetRWCat called with : " << std::get<1>(deltaStrat)->str() << "\n and " << std::get<2>(deltaStrat)->str();

        if (t==horizon-1){
            //std::cout << "\n strat last timestep : " << std::get<1>(deltaStrat)->str();

            std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> rwForLastTimestep = *(new std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double>());

            for (auto & tupleHistAction : std::get<1>(deltaStrat)->stratMap){

                for (auto & pairActionProba : tupleHistAction.second){

                    std::shared_ptr<HistoryTree> suffix = std::make_shared<HistoryTree>(std::make_shared<HistoryTree>(), std::pair(sdm::NO_OBSERVATION,pairActionProba.first));


                    std::shared_ptr<HistoryInterface> prefix = tupleHistAction.first;
                    rwForLastTimestep.emplace(std::tuple(prefix,suffix),pairActionProba.second);
                    
                }
            }
            //std::cout << " \n rwLastTimestep : ";
            /*
            for (auto & bla : rwForLastTimestep){
                std:: cout << "\n " << " prefix : " << std::get<0>(bla.first)->short_str() << " suffix : " << std::get<1>(bla.first)->short_str() << " val : " << bla.second;
            }*/
            return rwForLastTimestep;
        }

        //ELSE

        std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> rwNext = this->recGetRWMix(t+1,std::get<2>(deltaStrat),horizon);
        
        /*
        std::cout << "\n rwNext t=0 : ";
        for (auto & bla : rwNext){
                std:: cout << "\n " << " prefix : " << std::get<0>(bla.first)->short_str() << " suffix : " << std::get<1>(bla.first)->short_str() << " val : " << bla.second;
        }*/

        for (auto & pairPrefixSuffixValue : rwNext){

            //.pop for the prefix of pairPrefixSuffixValue
            std::shared_ptr<HistoryInterface> prefix = std::dynamic_pointer_cast<HistoryTree>(std::get<0>(pairPrefixSuffixValue.first))->getParent();

            //add to suffix
            const std::shared_ptr<HistoryInterface> suffix = std::get<1>(pairPrefixSuffixValue.first)->expand(std::get<0>(pairPrefixSuffixValue.first)->getLastObservation(),std::dynamic_pointer_cast<HistoryTree>(std::get<0>(pairPrefixSuffixValue.first))->getLastAction());

            rwCat[std::tuple(prefix,suffix)] = pairPrefixSuffixValue.second
                                                    *std::get<1>(deltaStrat)->getProbability(prefix,std::dynamic_pointer_cast<HistoryTree>(std::get<0>(pairPrefixSuffixValue.first))->getLastAction());
            
            
        }

        /*
        std::cout << "\n rwCat t=0 : ";
        for (auto & bla : rwCat){
                std:: cout << "\n " << " prefix : " << std::get<0>(bla.first)->short_str() << " suffix : " << std::get<1>(bla.first)->short_str() << " val : " << bla.second;
        }*/

        return rwCat;
    }


        ///////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////// Recursive get RW Mix //////////////////////////////////////////////

    std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> HsviMG::recGetRWMix(int t, const std::shared_ptr<VectorMG> & vector, int horizon){

        //std::cout << "\n vector in recGetRwMix : " << vector->str();
        //std::cout << "\n t : " << t;
        
        std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> res = *(new std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double>());

        for (auto & tupleW : vector->treeDeltaStrategies){//\forall w, get concatenation and add with proba
            if (tupleW.second>0.0){
                //std::cout << "\n call cat, vector : " << vector->str();

                std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> tmp = this->recGetRWCat(t,tupleW.first,horizon);//get concatenation for this w 
                
                //std::cout << "\n call done"<<std::flush;

                for (auto & rwToAdd : tmp){
                    if (res.count(rwToAdd.first)==0){
                        res.emplace(rwToAdd.first,tupleW.second*rwToAdd.second);
                    }

                    else{
                        res[rwToAdd.first] = res[rwToAdd.first] + tupleW.second*rwToAdd.second;
                    }
                    if (t==0){
                        //std::cout << "\n rw to add.first : " << std::dynamic_pointer_cast<HistoryTree>(std::get<0>(rwToAdd.first))->short_str();
                        //std::cout << "\n rw to add.second : " << std::dynamic_pointer_cast<HistoryTree>(std::get<1>(rwToAdd.first))->short_str();
                    }
                    if (t==0){
                    bool found = false;
                    for (auto & bla : rwForTimestep[t]){
                        if (std::dynamic_pointer_cast<HistoryTree>(std::get<0>(bla.first))->short_str() == std::dynamic_pointer_cast<HistoryTree>(std::get<0>(rwToAdd.first))->short_str()){
                            if (std::dynamic_pointer_cast<HistoryTree>(std::get<1>(bla.first))->short_str() == std::dynamic_pointer_cast<HistoryTree>(std::get<1>(rwToAdd.first))->short_str()){
                                if (!found){
                                this->rwForTimestep[t][bla.first] = 
                                    this->rwForTimestep[t][bla.first] + tupleW.second*rwToAdd.second;
                                    found = true;
                                }
                            }
                        }
                    }
                    if (!found){
                        this->rwForTimestep[t].emplace(rwToAdd.first,tupleW.second*rwToAdd.second);
                    }
                    }
                    else{
                        this->rwForTimestep[t][rwToAdd.first] = this->rwForTimestep[t][rwToAdd.first] + tupleW.second*rwToAdd.second;
                    }
                    //this->rwForTimestep[t].emplace(rwToAdd.first,tupleW.second*rwToAdd.second);
                    //this->rwForTimestep[t][rwToAdd.first] = this->rwForTimestep[t][rwToAdd.first] + tupleW.second*rwToAdd.second;
                }
            }   

        }
        
        /*
        std::cout << "\n rw : ";
        for (int i = 0;i<horizon;i++){
            std::cout << "\n at t :" << i;
            for (auto & bla : this->rwForTimestep[i]){
                std:: cout << "\n " << " prefix : " << std::get<0>(bla.first)->short_str() << " suffix : " << std::get<1>(bla.first)->short_str() << " val : " << bla.second;
            }
        }
        std::cout << "\n rw res : ";
        for (int i = 0;i<horizon;i++){
            std::cout << "\n at t :" << i;
            for (auto & bla : res){
                std:: cout << "\n " << " prefix : " << std::get<0>(bla.first)->short_str() << " suffix : " << std::get<1>(bla.first)->short_str() << " val : " << bla.second;
            }
        }
        std::cout << " \n end t : " << t;
        */
        return res;
    }

        ///////////////////////////////////////////////////////////////////////////////////////


    void HsviMG::initTrial()
    {
        // Do the pruning for the lower bound
        if (auto prunable_vf = std::dynamic_pointer_cast<PrunableStructure>(getLowerBound()))
            prunable_vf->doPruning(trial);

        // Do the pruning for the upper bound
        if (auto prunable_vf = std::dynamic_pointer_cast<PrunableStructure>(getUpperBound()))
            prunable_vf->doPruning(trial);
    }

    void HsviMG::initLogger()
    {
        // ************* Global Logger ****************
        // Text Format for standard output stream
        std::string format = "\r" + config::LOG_SDMS + "Trial {:<8} Error {:<12.7f} Value_LB {:<12.7f} Value_UB {:<12.7f} Size_LB {:<10} Size_UB {:<10} Time {:<12.4f}";

        // Titles of logs
        std::vector<std::string> list_logs{"Trial", "Error", "Value_LB", "Value_UB", "Size_LB", "Size_UB", "Time"};

        // Specific logs for belief MDPs
        if (sdm::isInstanceOf<BeliefMDPInterface>(getWorld()))
        {
            format = format + " NumState {:<8}";
            list_logs.push_back("NumState");
        }
        format = format + "";

        // Build a logger that prints logs on the standard output stream
        auto std_logger = std::make_shared<sdm::StdLogger>(format);

        // Build a logger that stores data in a CSV file
        auto csv_logger = std::make_shared<sdm::CSVLogger>(name, list_logs);

        // Build a multi logger that combines previous loggers
        this->logger = std::make_shared<sdm::MultiLogger>(std::vector<std::shared_ptr<Logger>>{std_logger, csv_logger});
    }

    void HsviMG::logging()
    {
        auto initial_state = getWorld()->getInitialState();

        if (auto derived = std::dynamic_pointer_cast<BeliefMDPInterface>(getWorld()))
        {
            // Print in loggers some execution variables
            logger->log(trial,
                        excess(initial_state, 0, 0) + error,
                        getLowerBound()->getValueAt(initial_state),
                        getUpperBound()->getValueAt(initial_state),
                        getLowerBound()->getSize(),
                        getUpperBound()->getSize(),
                        getExecutionTime(),
                        derived->getMDPGraph()->getNumNodes());
        }
        else
        {
            // Print in loggers some execution variables
            logger->log(trial,
                        excess(initial_state, 0, 0) + error,
                        getLowerBound()->getValueAt(initial_state),
                        getUpperBound()->getValueAt(initial_state),
                        getLowerBound()->getSize(),
                        getUpperBound()->getSize(),
                        getExecutionTime());
        }
    }

    std::string HsviMG::getAlgorithmName() { return "HsviMG"; }

    void HsviMG::saveParams(std::string filename, std::string format)
    {
        std::ofstream ofs;
        ofs.open(filename + format, std::ios::out | std::ios::app);

        if ((format == ".md"))
        {
            ofs << "## " << filename << "(PARAMS)" << std::endl;

            ofs << " | MAX_TRIAL | MAX_TIME | Error |  Horizon | p_o  | p_b | p_c | " << std::endl;
            ofs << " | --------- | -------- | ----- |  ------- | ---  | --- | --- | " << std::endl;
            ofs << " | " << num_max_trials;
            ofs << " | " << time_max;
            ofs << " | " << error;
            ofs << " | " << getWorld()->getHorizon();
            ofs << " | " << OccupancyState::PRECISION;
            ofs << " | " << Belief::PRECISION;
            ofs << " | " << PrivateOccupancyState::PRECISION_COMPRESSION;
            ofs << " | " << std::endl
                << std::endl;
        }
        ofs.close();
    }


    int HsviMG::testSolution(std::shared_ptr<VectorMG> vecUpb, std::shared_ptr<VectorMG> vecLob)
{
	std::string filename = "/home/delage/gitSdms3/sdms/data/world/dpomdp/matching_pennies.dpomdp";
	number horizon = 3, truncation = 0;
	double error = 0.0, discount = 1.;
    //std::cout << "\n vec : " << vecUpb->str();
    //std::cout << "\n vec : " << vecLob->str();
    //std::exit(1);
	try
	{
		int agent_id_=0;
		// Parse file into MPOMDP
		auto mdp = sdm::parser::parse_file(filename);
		mdp->setHorizon(horizon);
		mdp->setDiscount(discount);


		error=0.0;
		///////////////////////////////////// computing BR P2 ///////////////////////////////////////////
		
		//std::shared_ptr<VectorMG> vecUpb = std::dynamic_pointer_cast<partialQValueFunction>(ub)->getSupportVector(1);
		std::cout << "\n vec : " << vecUpb->str();
		
		std::shared_ptr<StochasticDecisionRule> extractedStrategy = this->extractStrategy(vecUpb,horizon,0);
        std::cout << extractedStrategy->str();
		extractedStrategy->testValidity();
		//std::cout << "\n extracted strategy : " << extractedStrategy->str();
		//std::exit(1);
		int solveBR = 1;
		if (solveBR){
			//now solve the BR 
			// Parse file into MPOMDP

			const number num_player_ = 0;
			// Instanciate the problem
			std::shared_ptr<SolvableByHSVI> hsvi_mdp_br = std::make_shared<PrivateOccupancyMDP>(mdp, extractedStrategy, num_player_,-1, true, true);

			// Instanciate Initializer
			auto lb_init_br = std::make_shared<MinInitializer>(hsvi_mdp_br);
			auto ub_init_br = std::make_shared<MaxInitializer>(hsvi_mdp_br);

			// Instanciate action selection 
			auto action_tabular_br = std::make_shared<ExhaustiveActionSelection>(hsvi_mdp_br);

			// Declare bounds
			std::shared_ptr<ValueFunction> lb_br, ub_br;

			// Instanciate lower bound
			lb_br = std::make_shared<TabularValueFunction>(hsvi_mdp_br, lb_init_br, action_tabular_br);

			// Instanciate lower bound update operator
			auto lb_update_operator_br = std::make_shared<TabularUpdate>(lb_br);
			lb_br->setUpdateOperator(lb_update_operator_br);

			// Instanciate upper bound
			ub_br = std::make_shared<TabularValueFunction>(hsvi_mdp_br, ub_init_br, action_tabular_br);

			// Instanciate upper bound update operator
			auto ub_update_operator_br = std::make_shared<TabularUpdate>(ub_br);
			ub_br->setUpdateOperator(ub_update_operator_br);
			
			// Instanciate HSVI
			auto algo_br = std::make_shared<HSVI>(hsvi_mdp_br, lb_br, ub_br, error, 10000, "", 1, 1);
			algo_br->agent_id_ = (num_player_*2)-1;
			
			//auto algo = std::make_shared<ValueIteration>(hsvi_mdp,lb,error,100000,"");

			// Initialize and solve the problem
			algo_br->initialize();
			algo_br->solve();
            double opt = algo_br->optimum;

            if (opt>vecUpb->getInitValue()+0.0001){
                std::cout<< "\n bug";
                std::cout << "\n upb : " << vecUpb->getInitValue() << "value BR : " << opt;
                std::cout << "\n strategy : " << extractedStrategy->str();
                std::exit(1);
            }
            //std::exit(1);
			//std::exit(1);
		}
		//std::exit(1);
		
		///////////////////////////////////// computing BR P1 ///////////////////////////////////////////
		
		//std::shared_ptr<VectorMG> vecLob = std::dynamic_pointer_cast<partialQValueFunction>(lb)->getSupportVector(0);
		std::cout << "\n vec : " << vecLob->str();
		
		std::shared_ptr<StochasticDecisionRule> extractedStrategyP1 = this->extractStrategy(vecLob,horizon,1);
        std::cout << extractedStrategyP1->str();
        extractedStrategyP1->testValidity();
		//std::cout << "\n extracted strategy player 2 : " << extractedStrategy->str();
		//std::cout << "\n extracted strategy player 1 : " << extractedStrategyP1->str();
		//std::exit(1);

		int solveBRP1 = 1;
		if (solveBRP1){
			//now solve the BR 
			// Parse file into MPOMDP

			const number num_player_ = 1;
			// Instanciate the problem
			std::shared_ptr<SolvableByHSVI> hsvi_mdp_br = std::make_shared<PrivateOccupancyMDP>(mdp, extractedStrategyP1, num_player_,-1, true, true);
			
			/*
			std::cout << "initial state : " << hsvi_mdp->initial_state_->str()<<std::flush;
			std::cout << "\naction space :  " << *mdp->getActionSpaceAt(0,0)<<std::endl;
			std::cout << "\ndr action space : " << *hsvi_mdp->getActionSpaceAt(hsvi_mdp->getInitialState(),0);
			*/

			// Instanciate Initializer
			auto lb_init_br = std::make_shared<MinInitializer>(hsvi_mdp_br);
			auto ub_init_br = std::make_shared<MaxInitializer>(hsvi_mdp_br);

			// Instanciate action selection 
			auto action_tabular_br = std::make_shared<ExhaustiveActionSelection>(hsvi_mdp_br);

			// Declare bounds
			std::shared_ptr<ValueFunction> lb_br, ub_br;

			// Instanciate lower bound
			lb_br = std::make_shared<TabularValueFunction>(hsvi_mdp_br, lb_init_br, action_tabular_br);

			// Instanciate lower bound update operator
			auto lb_update_operator_br = std::make_shared<TabularUpdate>(lb_br);
			lb_br->setUpdateOperator(lb_update_operator_br);

			// Instanciate upper bound
			ub_br = std::make_shared<TabularValueFunction>(hsvi_mdp_br, ub_init_br, action_tabular_br);

			// Instanciate upper bound update operator
			auto ub_update_operator_br = std::make_shared<TabularUpdate>(ub_br);
			ub_br->setUpdateOperator(ub_update_operator_br);
			
			// Instanciate HSVI
			auto algo_br = std::make_shared<HSVI>(hsvi_mdp_br, lb_br, ub_br, error, 10000, "", 1, 1);
			algo_br->agent_id_ = (num_player_*2)-1;
			
			//auto algo = std::make_shared<ValueIteration>(hsvi_mdp,lb,error,100000,"");

			// Initialize and solve the problem
			algo_br->initialize();
			algo_br->solve();

            double opt = -algo_br->optimum;
            if (opt<vecLob->getInitValue()-0.0001){
                std::cout<< "\n bug";
                std::cout << "\n upb : " << vecUpb->getInitValue() << "value BR : " << opt;
                std::cout << "\n strategy : " << extractedStrategyP1->str();
                std::exit(1);
            }
		}
	}
	catch (exception::Exception &e)
	{
		std::cout << "!!! Exception: " << e.what() << std::endl;
	}

	return 0;
} // END main
} // namespace sdm