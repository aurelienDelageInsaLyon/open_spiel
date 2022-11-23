#include <iomanip>
#include <sdm/config.hpp>
#include <sdm/core/state/occupancy_state_mg.hpp>
#include<sdm/world/pomdp.hpp>
#include<sdm/world/mpomdp.hpp>
#include <sdm/core/space/multi_discrete_space.hpp>
#include <sdm/world/registry.hpp>
#include <sdm/core/observation/default_observation.hpp>
#include <sdm/core/dynamics/tabular_state_dynamics.hpp>

namespace sdm
{

    double OccupancyStateMG::PRECISION_COMPRESSION = 0.0;//config::PRECISION_COMPRESSION;

    OccupancyStateMG::OccupancyStateMG() : OccupancyState()
    {
    }


    OccupancyStateMG::OccupancyStateMG(number num_agents, number h) : OccupancyState(num_agents, h)
    {
    }

    OccupancyStateMG::OccupancyStateMG(const OccupancyStateMG &v) 
    : OccupancyState(v)
    {

    }

    OccupancyStateMG::OccupancyStateMG(const OccupancyState &occupancy_state)
        : OccupancyState(occupancy_state)
    {
    }
    

    std::shared_ptr<OccupancyState> OccupancyStateMG::make(number h)
    {
        //std::cout << "this make is called";
        return std::make_shared<OccupancyStateMG>(this->num_agents_,h);
    }

    double OccupancyStateMG::getReward(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, number t)
    {
        return 0.0;
    }
    
    double OccupancyStateMG::norm_1_other(const std::shared_ptr<OccupancyStateMG> & other){
        double res = 0.0;

            for (auto o : this->getFullyUncompressedOccupancy()->getJointHistories())
            {
                res += std::abs(this->getProbability(o) - other->getProbability(o));
            }
            for (auto o : other->getFullyUncompressedOccupancy()->getJointHistories())
            {
                res += std::abs(this->getProbability(o) - other->getProbability(o));
                
            }
        return res;
    }

    Pair<std::shared_ptr<State>, double> OccupancyStateMG::computeNext(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, number t)
    {
        auto joint_dr = std::dynamic_pointer_cast<JointAction>(action);
        //std::cout << "\n act : " << act;
        //std:exit(1);
        // The new one step left occupancy state
        auto next_one_step_left_compressed_occupancy_state = this->make(t + 1);
        auto decision_rule = action->toDecisionRule();
        auto pomdp = std::dynamic_pointer_cast<POMDPInterface>(mdp);

        // For each joint history in the support of the fully uncompressed occupancy state
        for (const auto &compressed_joint_history : this->getJointHistories())
        {
            //std::cout << "\n jh : " << compressed_joint_history->str();
            //std::exit(1);
            // Get p(o_t)
            double proba_history = this->getProbability(compressed_joint_history);

            // Get the corresponding belief
            auto belief = this->getBeliefAt(compressed_joint_history);

            // Apply decision rule and get action
            //auto jaction = pomdp->getActionSpace()->sample()->toAction()->toJointAction();//this->applyDR(decision_rule, compressed_joint_history); // this->act(compressed_joint_history);

            // For each action that is likely to be taken
            for (const auto &joint_action : *pomdp->getActionSpace()) // decision_rule->getDistribution(compressed_joint_history)->getSupport())
            {
                // Get p(u_t | o_t)
                //std::cout<< "\n I'll ask for probability"<<"\n decision rule : " << typeid(decision_rule).name()<<std::flush;
                //double proba_action /*= 1; //*/= decision_rule->getProbability(compressed_joint_history, joint_action->toAction());
                auto stratP1 = std::dynamic_pointer_cast<StochasticDecisionRule>(joint_dr->get(0));
                auto stratP2 = std::dynamic_pointer_cast<StochasticDecisionRule>(joint_dr->get(1));
                
                double proba_action = stratP1->getProbability(compressed_joint_history->getIndividualHistory(0),joint_action->toAction()->toJointAction()->get(0))*stratP2->getProbability(compressed_joint_history->getIndividualHistory(1),joint_action->toAction()->toJointAction()->get(1));
                //std::cout << "\n proba_action : " << proba_action;
                //std::cout << "\n done" << std::flush;
                //std::exit(1);
                // For each observation in the space of joint observation
                for (auto jobs : *pomdp->getObservationSpace(t))
                {
                    auto joint_observation = jobs->toObservation();
                    if (this->checkCompatibility(joint_observation, observation))
                    {
                        // Get the next belief and p(z_{t+1} | b_t, u_t)
                        auto [next_belief, proba_observation] = belief->next(mdp, joint_action->toAction(), joint_observation, t);

                        double next_joint_history_probability = proba_history * proba_action * proba_observation;

                        // If the next history probability is not zero
                        if (next_joint_history_probability > 0)
                        {
                            // Update new one step uncompressed occupancy state
                            std::shared_ptr<JointHistoryInterface> next_compressed_joint_history = compressed_joint_history->expand(joint_observation, joint_action->toAction())->toJointHistory();
                            this->updateOccupancyStateProba(next_one_step_left_compressed_occupancy_state, next_compressed_joint_history, next_belief->toBelief(), next_joint_history_probability);
                        }
                    }
                }
            }
        }
        
        Pair<std::shared_ptr<State>, double> pair  = this->finalizeNextState(next_one_step_left_compressed_occupancy_state, t);

        auto occ_state = std::dynamic_pointer_cast<sdm::OccupancyState>(pair.first);
        std::shared_ptr<OccupancyStateMG> next_state = std::make_shared<OccupancyStateMG>(*occ_state);
        /*
        for (const auto &jhist : next_state->getJointHistories()){
            std::cout << "jhist : " << jhist->str();
        }*/

        //std::exit(1);
        return std::make_pair(next_state, pair.second);    
        //return this->finalizeNextState(next_one_step_left_compressed_occupancy_state, t);
    }
    /*
     Pair<std::shared_ptr<State>, double> OccupancyStateMG::computeNext(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, number t)
    {
        std::cout << "action : " << action->str() << "observation : " << observation->str();
        //std::exit(1);
        // The new one step left occupancy state
        auto next_one_step_left_compressed_occupancy_state = this->make(t+1);
        
        auto decision_rule = action->toDecisionRule();
        auto pomdp = std::dynamic_pointer_cast<POMDPInterface>(mdp);

        // For each joint history in the support of the fully uncompressed occupancy state
        for (const auto &compressed_joint_history : this->getJointHistories())
        {
            // Get p(o_t)
            double proba_history = this->getProbability(compressed_joint_history);

            // Get the corresponding belief
            auto belief = this->getBeliefAt(compressed_joint_history);

            // Apply decision rule and get action

            //NEEDS TO REDEFINE THIS
            //auto listJointActions = this->applyStochasticJointDR(decision_rule, compressed_joint_history); // this->act(compressed_joint_history);
            auto listJointActions= pomdp->getActionSpace();
            // For each action that is likely to be taken
            for (const auto &joint_action : *listJointActions) // decision_rule->getDistribution(compressed_joint_history)->getSupport())
            {
                // Get p(u_t | o_t)

                //NEEDS TO REDEFINE THIS
                //double proba_action = 1; // decision_rule->getProbability(compressed_joint_history, joint_action);
                double proba_action = decision_rule->getProbability(compressed_joint_history,joint_action->toAction());
                // For each observation in the space of joint observation
                for (auto jobs : *pomdp->getObservationSpace(t))
                {
                    auto joint_observation = jobs->toObservation();
                    auto j_act = joint_action->toAction();
                    if (this->checkCompatibility(joint_observation, observation))
                    {
                        // Get the next belief and p(z_{t+1} | b_t, u_t)
                        auto [next_belief, proba_observation] = belief->next(mdp, joint_action->toAction(), joint_observation, t);
                        std::cout << "\n next belief for action : " << joint_action->toAction()->str() << " and observations : " << joint_observation->str() << next_belief->str();

                        double next_joint_history_probability = proba_history * proba_action * proba_observation;

                        // If the next history probability is not zero
                        if (next_joint_history_probability > 0)
                        {
                            std::cout << "\n pointer hist null : " << compressed_joint_history->str();
                            std::cout << "proba next joint_history : " << next_joint_history_probability;
                            // Update new one step uncompressed occupancy state
                            std::cout << "\n joint observation : " << joint_observation->str() << " joint action : " << joint_action->str();
                            std::shared_ptr<JointHistoryInterface> next_compressed_joint_history = compressed_joint_history->expand(joint_observation,j_act)->toJointHistory();
                            std::cout << "\n compressed_jh : " << next_compressed_joint_history->str();
                            this->updateOccupancyStateProba(next_one_step_left_compressed_occupancy_state, next_compressed_joint_history, next_belief->toBelief(), next_joint_history_probability);
                        }
                    }
                }
                                    std::exit(1);

            }
        }
        //return std::make_pair(next_one_step_left_compressed_occupancy_state,t);
        Pair<std::shared_ptr<State>, double> pair  = this->finalizeNextState(next_one_step_left_compressed_occupancy_state, t);

        auto occ_state = std::dynamic_pointer_cast<sdm::OccupancyState>(pair.first);
        std::shared_ptr<OccupancyStateMG> next_state = std::make_shared<OccupancyStateMG>(*occ_state);
        
        for (const auto &jhist : next_state->getJointHistories()){
            std::cout << "jhist : " << jhist->str();
        }

        //std::exit(1);
        return std::make_pair(next_state, pair.second);    
        }
        */
}