#include <iomanip>
#include <sdm/config.hpp>
#include <sdm/core/state/private_br_occupancy_state.hpp>
#include<sdm/world/pomdp.hpp>
#include<sdm/world/mpomdp.hpp>

namespace sdm
{

    double PrivateBrOccupancyState::PRECISION_COMPRESSION = config::PRECISION_COMPRESSION;

    PrivateBrOccupancyState::PrivateBrOccupancyState()
    {
    }

    PrivateBrOccupancyState::PrivateBrOccupancyState(number num_agents, number h, StochasticDecisionRule dr) : OccupancyState(num_agents, h), strategyOther(dr)
    {
    }

    PrivateBrOccupancyState::PrivateBrOccupancyState(number agent_id, number num_agents, number h, StochasticDecisionRule dr) : OccupancyState(num_agents, h), agent_id_(agent_id), strategyOther(dr)
    {
    }


    PrivateBrOccupancyState::PrivateBrOccupancyState(number agent_id, number num_agents, number h) : OccupancyState(num_agents, h), agent_id_(agent_id)
    {
    }

    PrivateBrOccupancyState::PrivateBrOccupancyState(const PrivateBrOccupancyState &v)
        : OccupancyState(v),
          agent_id_(v.getAgentId()),
          map_jhist_to_partial(v.map_jhist_to_partial),
          map_partial_to_jhist(v.map_partial_to_jhist)
    {
    }

    PrivateBrOccupancyState::PrivateBrOccupancyState(const OccupancyState &occupancy_state)
        : OccupancyState(occupancy_state),
          agent_id_(0)
    {
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
        res << "</private-occupancy-state>";
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
        //the agent i (computing the br);
        number i = 1;

        // The new one step left occupancy state
        auto next_one_step_left_compressed_occupancy_state = this->make(t + 1);
        auto decision_rule = action->toDecisionRule();
        auto pomdp = std::dynamic_pointer_cast<MPOMDP>(mdp);

        // For each joint history in the support of the fully uncompressed occupancy state
        for (const auto &compressed_joint_history : this->getJointHistories())
        {   
            auto act = this->applyDR(decision_rule, compressed_joint_history);//to change ! 
            
            // Get p(o_t)
            double proba_history = this->getProbability(compressed_joint_history);

            // Get the corresponding belief b_t(s_t,\theta_t^i)
            auto belief = this->getBeliefAt(compressed_joint_history);
            
            for (const auto &action_br : {act}) //to change
            {
                for (const auto &action_other : this->strategyOther.getActions(compressed_joint_history)){
                {
                    double proba_action = this->strategyOther.getProbability(compressed_joint_history,action_other);

                    for (auto jobs : *pomdp->getActionSpace(i,t)){

                    //à demander : je ne sais même pas comment récupérer une observation privée !
                    auto joint_observation = jobs->toObservation();
                    
                    //overrides the belief computation !
                    //get the probability of joint observation (obs,jobs) (avec jobs = {none,obs}).
                    //and obs is the joint observation, so that one can normalize the final belief b(x,h)
                    //to get the probability of this public observation and a valid belief b(x,h)
                    auto [next_belief, proba_observation] = this->getNewBelief(belief,action,observation,joint_observation,t,mdp);


                        double next_joint_history_probability = proba_history * proba_action * proba_observation;

                        // If the next history probability is not zero
                        if (next_joint_history_probability > 0)
                        {
                            // Update new one step uncompressed occupancy state
                            //this is to be changed !
                            /*
                            std::shared_ptr<JointHistoryInterface> next_compressed_joint_history = compressed_joint_history->expand(joint_observation)->toJointHistory();
                            this->updateOccupancyStateProba(next_one_step_left_compressed_occupancy_state, next_compressed_joint_history, next_belief->toBelief(), next_joint_history_probability);
                            */

                           //à demander : pourquoi expand(joint_observation) et pas expand(joint_observation,action) ??
                           std::shared_ptr<JointHistoryInterface> next_compressed_joint_history = compressed_joint_history->expand(joint_observation)->toJointHistory();
                           this->updateOccupancyStateProba(next_one_step_left_compressed_occupancy_state, next_compressed_joint_history, next_belief->toBelief(), next_joint_history_probability);
                        }
                        }
                    }
                }
                //on est bien d'accord qu'on récupère P(z^{\neg i}) là, non?
                double probaPublicObs = belief->norm_1();
                belief->normalizeBelief(probaPublicObs);

            }

        }

        return this->finalizeNextState(next_one_step_left_compressed_occupancy_state, t);
    }
    
    Pair<std::shared_ptr<BeliefInterface>, double> PrivateBrOccupancyState::getNewBelief(const std::shared_ptr<BeliefInterface> &belief, const std::shared_ptr<Action> &action, const std::shared_ptr<Observation> &observation, const std::shared_ptr<Observation> & jobs, number t, const std::shared_ptr<MDPInterface> &mdp) const
    {
    
    std::shared_ptr<BeliefInterface> nextBelief;
    double proba=1.0;
    auto pomdp = std::dynamic_pointer_cast<POMDPInterface>(mdp);

    // Create next belief.
    //auto pomdp = std::dynamic_pointer_cast<POMDPInterface>(mdp);
    for (const auto &old_state : belief->getStates())
    {
      for (const auto &next_state : belief->getStates())
      {
        double proba = pomdp->getDynamics(old_state, action, next_state, observation, t) * belief->getProbability(old_state) ;

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
} // namespace sdm
