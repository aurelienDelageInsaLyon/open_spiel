#include <iomanip>
#include <sdm/config.hpp>
#include <sdm/core/state/private_br_occupancy_state.hpp>
#include<sdm/world/pomdp.hpp>
#include<sdm/world/mpomdp.hpp>
#include <sdm/core/space/multi_discrete_space.hpp>
#include <sdm/world/registry.hpp>
#include <sdm/core/observation/default_observation.hpp>
#include <sdm/core/dynamics/tabular_state_dynamics.hpp>

namespace sdm
{

    double PrivateBrOccupancyState::PRECISION_COMPRESSION = config::PRECISION_COMPRESSION;

    PrivateBrOccupancyState::PrivateBrOccupancyState() : OccupancyState()
    {
        this->list_joint_histories_= std::set<std::shared_ptr<sdm::JointHistoryInterface>>();
    }

    PrivateBrOccupancyState::PrivateBrOccupancyState(number num_agents, number h, StochasticDecisionRule dr) : OccupancyState(num_agents, h), strategyOther(dr)
    {
    }

    PrivateBrOccupancyState::PrivateBrOccupancyState(number agent_id, number num_agents, number h, StochasticDecisionRule dr) : OccupancyState(num_agents, h), agent_id_(agent_id), strategyOther(dr)
    {
        //std::cout << "\n putting strat : " << strategyOther.str();
    }


    PrivateBrOccupancyState::PrivateBrOccupancyState(number agent_id, number num_agents, number h) : OccupancyState(num_agents, h), agent_id_(agent_id)
    {
    }

    PrivateBrOccupancyState::PrivateBrOccupancyState(const PrivateBrOccupancyState &v) 
        : OccupancyState(v),
          agent_id_(v.agent_id_),
          strategyOther(v.strategyOther)
    {

    }

    PrivateBrOccupancyState::PrivateBrOccupancyState(const OccupancyState &occupancy_state)
        : OccupancyState(occupancy_state),
          agent_id_(0)
    {
        std::cout << "\n privatebrocc::this should never be called. I'm exiting.";
        std::exit(1);
    }
    
    PrivateBrOccupancyState::PrivateBrOccupancyState(const OccupancyState &occupancy_state,number id, StochasticDecisionRule dr)
        : OccupancyState(occupancy_state),
          agent_id_(id),
          strategyOther(dr)
    {

    }

    std::shared_ptr<OccupancyState> PrivateBrOccupancyState::make(number h)
    {
        return std::make_shared<PrivateBrOccupancyState>(this->num_agents_, h,this->strategyOther);
    }

    number PrivateBrOccupancyState::getAgentId() const
    {
        return this->agent_id_;
    }

    std::string PrivateBrOccupancyState::str() const
    {
        std::ostringstream res;
        res << std::setprecision(config::OCCUPANCY_DECIMAL_PRINT) << std::fixed;

        res << "<private-occupancy-state agent=\"" << this->agent_id_ << "\" size=\"" << this->size() << "\">\n";
        for (const auto &history_as_state : this->getStates())
        {
            auto joint_history = history_as_state->toHistory()->toJointHistory();

            res << "\t<probability";
            res << " history=" << joint_history->short_str() << "";
            res << " belief=" << this->getBeliefAt(joint_history)->str() << ">\n";
            res << "\t\t\t" << this->getProbability(joint_history) << "\n";
            res << "\t</probability \n";
        }
        res << "\n <agent_id_> " << this->agent_id_;
        //res << "\n <base actions> " << this->action_space_map->at(1);
        res << "</private-occupancy-state/>";
        return res.str();
    }

    std::vector<std::shared_ptr<HistoryInterface>> PrivateBrOccupancyState::getPartialJointHistory(const std::vector<std::shared_ptr<HistoryInterface>> &joint_history) const
    {
        // Copy full joint history
        auto partial_jhist = joint_history;

        // Erase the component associated to the agent
        partial_jhist.erase(partial_jhist.begin() + this->getAgentId());

        // Return the partial joint history
        return partial_jhist;
    }

    const std::vector<std::shared_ptr<HistoryInterface>> &PrivateBrOccupancyState::getPartialJointHistory(const std::shared_ptr<JointHistoryInterface> &joint_history) const
    {
        return this->map_jhist_to_partial.at(joint_history);
    }

    std::shared_ptr<JointHistoryInterface> PrivateBrOccupancyState::getJointHistoryFromPartial(const std::vector<std::shared_ptr<HistoryInterface>> &partial_joint_history) const
    {
        return this->map_partial_to_jhist.at(partial_joint_history);
    }

    void PrivateBrOccupancyState::finalize()
    {
        OccupancyState::finalize();
    }

    void PrivateBrOccupancyState::finalize(bool do_compression)
    {
        if (do_compression)
        {
            this->finalize();
        }
        else
        {
            this->setup();
        }

        // Add elements in bimap jhistory <--> jhistory^{-i}
        for (const auto &joint_history : this->getJointHistories())
        {
            // Get partial joint history
            const auto &partial_jhist = this->getPartialJointHistory(joint_history->getIndividualHistories());
            //
            this->map_partial_to_jhist[partial_jhist] = joint_history;
            this->map_jhist_to_partial[joint_history] = partial_jhist;
        }
    }

    bool PrivateBrOccupancyState::check_equivalence(const PrivateBrOccupancyState &other) const
    {
        // Check that private occupancy states are defined on the same support
        if (this->size() != other.size())
        {
            return false;
        }
        // Go over all partial joint histories and associated joint history
        for (const auto &pair_partial_jhistory : this->map_partial_to_jhist)
        {
            const auto &partial_joint_history = pair_partial_jhistory.first;
            const auto &current_joint_history = pair_partial_jhistory.second;

            // Get an iterator on the first partial joint history that is similar in "other"
            auto iterator = other.map_partial_to_jhist.find(partial_joint_history);
            if (iterator == other.map_partial_to_jhist.end())
            {
                return false;
            }

            // Get the associated joint history of the second value
            const auto &other_joint_history = iterator->second;

            // For all states in the corresponding belief
            for (const auto &state : this->getBeliefAt(current_joint_history)->getStates())
            {
                // Compare p(o^{-i}, x | o^{i}_1) and p(o^{-i}, x | o^{i}_2)
                if (std::abs(this->getProbability(current_joint_history, state) - other.getProbability(other_joint_history, state)) > PrivateBrOccupancyState::PRECISION_COMPRESSION)
                {
                    return false;
                }
            }
        }
        return true;
    }

    std::shared_ptr<Action> PrivateBrOccupancyState::applyDR(const std::shared_ptr<DecisionRule> &dr, const std::shared_ptr<HistoryInterface> &individual_history) const
    {
        return dr->act(individual_history);
    }

    std::shared_ptr<Action> PrivateBrOccupancyState::applyStochasticDR(const StochasticDecisionRule dr, const std::shared_ptr<HistoryInterface> &individual_history) const
    {
        StochasticDecisionRule stochasticDr = (StochasticDecisionRule) dr;
        return stochasticDr.act(individual_history);
    }
    //REDEFINED
    //method that computes the next occupancy state according to a (public) observation
    //public observation <-> \theta^\neg i avec $i$ qui calcule une BR à \beta^i, non?
       Pair<std::shared_ptr<State>, double> PrivateBrOccupancyState::computeNext(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, number t)
    {

        this->reward = this->getReward(mdp,action,t);        
        // The new one step left occupancy state
        auto next_one_step_left_compressed_occupancy_state = this->make(t+1);

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
                    //std::cout << "\n compressed jh : " << compressed_joint_history->toJointHistory()->getIndividualHistory((this->agent_id_+1)%2)->str();
                    double proba_action = this->strategyOther.getProbability(compressed_joint_history->toJointHistory()->getIndividualHistory((this->agent_id_+1)%2),action_other->toAction());

                    // MODIFIER, jobs doit parcourir getObservaitonSpace(this->agent_id+1%2)!!!
                    //std::exit(1);
                    for (auto jobs : *pomdp->getObservationSpace((this->agent_id_+1)%2,t)){

                    auto jact = this->getActionPointer(action->toAction(),action_other->toAction(),pomdp,this->agent_id_);
                    
                    auto joint_obs = this->getObservationPointer(observation,jobs->toObservation(),pomdp,
                    this->agent_id_);

                    auto [next_belief, proba_observation] = belief->next(mdp, jact->toAction(), joint_obs->toObservation(), t);

                    //proba_observation is the probability to get the joint observation
                        double next_joint_history_probability = proba_history * proba_action * proba_observation;

                        // If the next history probability is not zero
                        if (next_joint_history_probability > 0)
                        {

                           std::shared_ptr<JointHistoryInterface> next_compressed_joint_history = compressed_joint_history->expand(joint_obs->toObservation(), jact->toAction())->toJointHistory();
                           //std::cout << "\n nest hist: " << next_compressed_joint_history->str();
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
    }
    
    void PrivateBrOccupancyState::updateOccupancyStateProba(const std::shared_ptr<OccupancyStateInterface> &occupancy_state, const std::shared_ptr<JointHistoryInterface> &joint_history, const std::shared_ptr<BeliefInterface> &belief, double probability)
    {
        //for compression/LPE compression !!
        if (occupancy_state->getProbability(joint_history) > 0.)
        {
            // Get the probability of being in each belief
            double proba_belief1 = occupancy_state->getProbability(joint_history), proba_belief2 = probability;
            // Cast to belief structure
            std::shared_ptr<Belief> belief1 = std::dynamic_pointer_cast<Belief>(occupancy_state->getBeliefAt(joint_history)), belief2 = std::dynamic_pointer_cast<Belief>(belief);

            // Aggregate beliefs
            std::shared_ptr<Belief> aggregated_belief = std::make_shared<Belief>(belief1->add(*belief2, proba_belief1, proba_belief2));

            // Normalize the resulting belief
            aggregated_belief->normalizeBelief(aggregated_belief->norm_1());

            // Build fully uncompressed occupancy state
            occupancy_state->setProbability(joint_history, aggregated_belief, occupancy_state->getProbability(joint_history) + probability);
        }
        else
        {// this is what is interessting
            // Build fully uncompressed occupancy state
            occupancy_state->setProbability(joint_history, belief, probability);
        }
    }

    std::shared_ptr<sdm::JointAction> PrivateBrOccupancyState::getActionPointer(std::shared_ptr<sdm::Action> a, std::shared_ptr<sdm::Action> b,
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
        return nullptr;
    }

    std::shared_ptr<sdm::Observation> PrivateBrOccupancyState::getObservationPointer(std::shared_ptr<sdm::Observation> a, std::shared_ptr<sdm::Observation> b,
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


    double PrivateBrOccupancyState::getReward(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, number t)
    {
        //std::cout<< "\n i'm in get reward from occupancy_state !" << std::flush << std::endl;
        double reward = 0.;
        
        auto pomdp = std::dynamic_pointer_cast<MPOMDP>(mdp);

        //auto decision_rule = action->toDecisionRule();
        // For all histories in the occupancy state
        for (const auto &jhist : this->getJointHistories())
        {   
        //std::cout << "\n jhist : " << jhist->str();

            auto listActions = std::dynamic_pointer_cast<MultiDiscreteSpace>(pomdp->getActionSpace());
            // For each joint history in the support of the fully uncompressed occupancy state
            auto listActionsCasted = std::dynamic_pointer_cast<DiscreteSpace>(listActions->getSpace((this->agent_id_+1)%2));
            this->strategyOther.action_space = *listActionsCasted;
            this->strategyOther.numActions = listActionsCasted->getNumItems();
            // Get the belief corresponding to this history
            auto belief = this->getBeliefAt(jhist);
            //std::cout << "\n getreward::belief : " << belief->str();
            //std::exit(1);
            // Get the action from decision rule
            //auto joint_action = this->applyDR(decision_rule, jhist); // decision_rule->act(jhist);
            for (auto & action_other : *pomdp->getActionSpace((this->agent_id_+1)%2,t)){
                auto joint_action = this->getActionPointer(action->toAction(),action_other->toAction(),pomdp, this-> agent_id_);
                //double proba = this->strategyOther.getProbability(jhist->getIndividualHistory(this->agent_id_),action_other->toAction());
                double proba = this->strategyOther.getProbability(jhist->getIndividualHistory((this->agent_id_+1)%2),action_other->toAction());
                //std::cout << "\n jhist : " << jhist->str() << "proba jhist : " <<this->getProbability(jhist);
                //PB ICI !
                //std::cout << "\n belief : " << this->str();
                //std::cout << "\n proba jhist : " << this->getProbabilityOverIndividualHistories(this->agent_id_,//jhist->getIndividualHistory(this->agent_id_));
                //std::cout << "\n belief->getReward() : " << belief->getReward(mdp, joint_action, t);
                //std::cout << " for actions : " << joint_action->str();
                // Update the expected reward
                reward += proba*this->getProbability(jhist) * belief->getReward(mdp, joint_action, t);
                //std::cout << "\n t : " << t << "reward here : " << reward*4/10 << std::endl;
            }
        
        //std::cout<< "\n i'm leaving get reward from private_br_occupancy_state !" << std::flush << std::endl;
        //std::cout << "\n belief : " << this->str();
        //std::cout << "computed reward : " << reward*4/10 << " at timestep : " << t << " for action : " << action->str();
        }
        if (this->agent_id_==0){
            return reward;
        }
        else{
            return -reward;
        }
        //return reward;
    }

    double PrivateBrOccupancyState::norm_1_other(const std::shared_ptr<OccupancyState> & other){
        double res = 0.0;
        std::unordered_set<std::shared_ptr<HistoryInterface>> setAlreadyComputed;

        if (false && other->list_joint_histories_.size()==0){

        for (auto & h : this->getJointHistories()){
            res += std::abs(this->getProbability(h));
            setAlreadyComputed.emplace(h);
        }
        return res;
        }
        for (auto & h : this->getJointHistories()){
            res += std::abs(this->getProbability(h) - other->getProbability(h));
            setAlreadyComputed.emplace(h);
        }

        for (auto & h : other->getJointHistories()){
            if (setAlreadyComputed.find(h) ==setAlreadyComputed.end()){
                res += std::abs(this->getProbability(h) - other->getProbability(h));
            }
        }
        return res;
    }


/*
       Pair<std::shared_ptr<State>, double> OccupancyState::computeNext(const std::shared_ptr<MDPInterface> &mdp, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, number t)
    {
        // The new one step left occupancy state
        auto next_one_step_left_compressed_occupancy_state = this->make(t + 1);
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
            auto jaction = this->applyDR(decision_rule, compressed_joint_history); // this->act(compressed_joint_history);

            // For each action that is likely to be taken
            for (const auto &joint_action : {jaction}) // decision_rule->getDistribution(compressed_joint_history)->getSupport())
            {
                // Get p(u_t | o_t)
                double proba_action = 1; // decision_rule->getProbability(compressed_joint_history, joint_action);

                // For each observation in the space of joint observation
                for (auto jobs : *pomdp->getObservationSpace(t))
                {
                    auto joint_observation = jobs->toObservation();
                    if (this->checkCompatibility(joint_observation, observation))
                    {
                        // Get the next belief and p(z_{t+1} | b_t, u_t)
                        auto [next_belief, proba_observation] = belief->next(mdp, joint_action, joint_observation, t);

                        double next_joint_history_probability = proba_history * proba_action * proba_observation;

                        // If the next history probability is not zero
                        if (next_joint_history_probability > 0)
                        {
                            // Update new one step uncompressed occupancy state
                            std::shared_ptr<JointHistoryInterface> next_compressed_joint_history = compressed_joint_history->expand(joint_observation )->toJointHistory();
                            this->updateOccupancyStateProba(next_one_step_left_compressed_occupancy_state, next_compressed_joint_history, next_belief->toBelief(), next_joint_history_probability);
                        }
                    }
                }
            }
        }

        return this->finalizeNextState(next_one_step_left_compressed_occupancy_state, t);
    }
    */
   /*
    Pair<std::shared_ptr<BeliefInterface>, double> PrivateBrOccupancyState::getNewBelief(const std::shared_ptr<BeliefInterface> &belief, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, const std::shared_ptr<Observation> & jobs, number t, const std::shared_ptr<MDPInterface> &mdp) const
    {
    
    //std::cout << "\n <computing the next belief for action, observation : >" << action->str() << "," << //observation->str() << "," << jobs->str()<<"\n";
    //std::exit(1);
    std::shared_ptr<BeliefInterface> nextBelief;
    double proba=1.0;
    auto pomdp = std::dynamic_pointer_cast<POMDPInterface>(mdp);

    //std::cout << "\n is pomdp null ?" << pomdp;
    // Create next belief.
    //auto pomdp = std::dynamic_pointer_cast<POMDPInterface>(mdp);
    auto joint_observation = sdm::JointObservation({observation->toObservation(),jobs->toObservation()});
    auto joint_observation_ptr = std::make_shared<sdm::JointObservation>(joint_observation);
    auto joint_obs = joint_observation_ptr->toObservation();
    for (const auto &old_state : belief->getStates())
    {
      for (const auto &next_state : belief->getStates())
      {
        double proba = pomdp->getDynamics(old_state, action, next_state, joint_obs, t) * belief->getProbability(old_state) ;

        std::cout << "proba : " << proba;

        if (proba > 0)
        {
          nextBelief->addProbability(next_state, proba);
        }
      }
    }
    nextBelief->finalize();

    // Compute the coefficient of normalization (eta)
    double eta = nextBelief->norm_1();

    // Normalize to belief
    nextBelief->normalizeBelief(eta);


    return std::make_pair(nextBelief,proba);
    }
    */
   //
   //
   /*
    std::shared_ptr<sdm::Action> PrivateBrOccupancyState::getIndivActionPointer(std::shared_ptr<sdm::Action> a,
    std::shared_ptr<POMDPInterface> pomdp) const
    {
        for (const auto &joint_action : *pomdp->getActionSpace()){
            if (joint_action->toAction()->toJointAction()->get(this->agent_id_)->str()==a->str()){
                return joint_action->toAction();
            }
        }
        return nullptr;
    }

    std::shared_ptr<sdm::Observation> PrivateBrOccupancyState::getIndivObservationPointer(std::shared_ptr<sdm::Observation> a,
    std::shared_ptr<POMDPInterface> pomdp) const
    {
        for (const auto &joint_observation : *pomdp->getObservationSpace(0)){
            //std::cout << "\n is it a joint obs? " << std::dynamic_pointer_cast<JointItem>(joint_observation);
            
            if (std::dynamic_pointer_cast<JointItem>(joint_observation)->get(this->agent_id_)->str()==a->str()){
                return joint_observation->toObservation();
            }
        }
        return nullptr;
    }
    */
} // namespace sdm
