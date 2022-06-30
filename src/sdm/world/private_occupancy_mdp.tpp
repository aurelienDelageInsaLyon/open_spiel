#include <memory>
#include <sdm/world/private_occupancy_mdp.hpp>
#include <sdm/world/registry.hpp>

namespace sdm
{
    template <class TOccupancyState>
    BasePrivateOccupancyMDP<TOccupancyState>::BasePrivateOccupancyMDP()
    {
    }

    template <class TOccupancyState>
    BasePrivateOccupancyMDP<TOccupancyState>::~BasePrivateOccupancyMDP()
    {
    }

    template <class TOccupancyState>
    BasePrivateOccupancyMDP<TOccupancyState>::BasePrivateOccupancyMDP(const std::shared_ptr<MPOMDPInterface> &decpomdp, const number num_player_, int memory, bool store_states, bool store_actions, int batch_size)
        : decpomdp(decpomdp), memory(memory)
    {
        this->store_states_ = store_states;
        this->store_actions_ = store_actions;
        this->batch_size_ = batch_size;
        this->mdp = decpomdp;

        // Initialize underlying belief mdp
        this->belief_mdp_ = std::make_shared<BeliefMDP>(decpomdp, batch_size, store_states, store_actions);

        // Initialize initial history
        this->initial_history_ = std::make_shared<JointHistoryTree>(this->mdp->getNumAgents(), this->memory);

        // Initialize initial occupancy state
        //TODO modifier pour rendre compatible avec factored_occupancy_state, i.e., en argument on trouve le modele decpomdp ou ndpomdp plus peut-etre la memoire.

        this->initial_state_ = (std::shared_ptr<PrivateBrOccupancyState>) std::make_shared<PrivateBrOccupancyState>(this->mdp->getNumAgents(), 0,0);
        this->initial_state_->agent_id_ = -1;

        this->initial_state_->toOccupancyState()->setProbability(this->initial_history_->toJointHistory(), this->belief_mdp_->getInitialState()->toBelief(), 1);
        this->initial_state_->toOccupancyState()->finalize();
        
        // Initialize Transition Graph
        this->mdp_graph_ = std::make_shared<Graph<std::shared_ptr<State>, Pair<std::shared_ptr<Action>, std::shared_ptr<Observation>>>>();
        this->mdp_graph_->addNode(this->initial_state_);

        this->reward_graph_ = std::make_shared<Graph<double, Pair<std::shared_ptr<State>, std::shared_ptr<Action>>>>();
        this->reward_graph_->addNode(0.0);
        
        this->num_player_ = num_player_;

    }

    template <class TOccupancyState>
    BasePrivateOccupancyMDP<TOccupancyState>::BasePrivateOccupancyMDP(const std::shared_ptr<MPOMDPInterface> &decpomdp, const number num_player_, Config config)
        : BasePrivateOccupancyMDP<TOccupancyState>(decpomdp,
                                            config.get("memory", 0),
                                            config.get("store_states", true),
                                            config.get("store_actions", true),
                                            config.get("batch_size", 0))
    {
        auto opt_int = config.getOpt<int>("state_type");
        auto opt_str = config.getOpt<std::string>("state_type");
        if (opt_int.has_value())
            this->setStateType((StateType)opt_int.value());
        else if (opt_str.has_value())
        {
            auto iter = STATE_TYPE_MAP.find(opt_str.value());
            this->setStateType((iter != STATE_TYPE_MAP.end()) ? iter->second : StateType::COMPRESSED);
        }
        this->num_player_ = num_player_;
    }

    template <class TOccupancyState>
    BasePrivateOccupancyMDP<TOccupancyState>::BasePrivateOccupancyMDP(Config config)
        : BasePrivateOccupancyMDP<TOccupancyState>(std::dynamic_pointer_cast<MPOMDPInterface>(sdm::world::createFromConfig(config)), config)
    {
    }


    //REDEFINED -> to return as public observations all the observation of player num_player_
    template <class TOccupancyState>
    std::shared_ptr<Space> BasePrivateOccupancyMDP<TOccupancyState>::getObservationSpaceAt(const std::shared_ptr<State> &, const std::shared_ptr<Action> &, number t)
    {
        return this->decpomdp->getObservationSpace(this->num_player_,t);
    }

    template <class TOccupancyState>
    void BasePrivateOccupancyMDP<TOccupancyState>::setStateType(const StateType &state_type)
    {
        this->state_type = state_type;
        this->initial_state_->toOccupancyState()->setStateType(state_type);
    }

    template <class TOccupancyState>
    bool BasePrivateOccupancyMDP<TOccupancyState>::checkCompatibility(const std::shared_ptr<Observation> &, const std::shared_ptr<Observation> &)
    {
        return true;
    }

    template <class TOccupancyState>
    std::shared_ptr<MPOMDPInterface> BasePrivateOccupancyMDP<TOccupancyState>::getUnderlyingMPOMDP() const
    {
        return this->decpomdp;
    }

    template <class TOccupancyState>
    std::shared_ptr<BeliefMDPInterface> BasePrivateOccupancyMDP<TOccupancyState>::getUnderlyingBeliefMDP()
    {
        return this->belief_mdp_;
    }

    template <class TOccupancyState>
    double BasePrivateOccupancyMDP<TOccupancyState>::do_excess(double incumbent, double lb, double ub, double cost_so_far, double error, number horizon)
    {
        // return std::min(ub - lb, cost_so_far + this->mdp->getDiscount(horizon) * ub - incumbent) - error / this->getWeightedDiscount(horizon);
        return (ub - lb) - error / this->getWeightedDiscount(horizon);
    }

    // -------------------
    //     RL METHODS
    // -------------------

    template <class TOccupancyState>
    std::shared_ptr<State> BasePrivateOccupancyMDP<TOccupancyState>::reset()
    {
        return BaseBeliefMDP<TOccupancyState>::reset();
    }

    template <class TOccupancyState>
    std::tuple<std::shared_ptr<State>, std::vector<double>, bool> BasePrivateOccupancyMDP<TOccupancyState>::step(std::shared_ptr<Action> action)
    {
        // Compute next reward
        //needs to be redefined to include opponent's strategy because here, action is single action
        double occupancy_reward = this->getReward(this->current_state_, action, this->step_);


        // Compute next occupancy state
        this->current_state_ = this->getNextStateAndProba(this->current_state_, action, sdm::NO_OBSERVATION, this->step_).first;

        // Increment step
        this->step_++;

        bool is_done = (this->getHorizon() > 0) ? (this->step_ >= this->getHorizon()) : false;
        return std::make_tuple(this->current_state_, std::vector<double>(this->mdp->getNumAgents(), occupancy_reward), is_done);
    }

    // -----------------------
    // Manipulate actions
    // -------------------------


    //method that indicates to the algorithm using an occupancy_mdp which action are available for a particular
    //occupancy_state ostate at timestep t.

    //Redefined -> to return only the (pure) decision rules for the best-response player.

    template <class TOccupancyState>
    std::shared_ptr<Space> BasePrivateOccupancyMDP<TOccupancyState>::getActionSpaceAt(const std::shared_ptr<State> &ostate, number t)
    {
        auto occupancy_state = ostate->toOccupancyState();
        // If the action space corresponding to this ostate and t does not exist:
        if (occupancy_state->getActionSpaceAt(t) == nullptr)
        {
            // Compute the action space at this occupancy state and timestep
            std::shared_ptr<Space> joint_ddr_space = this->computeActionSpaceAt(ostate, t);

            if (!this->store_actions_)
            {
                return joint_ddr_space;
            }

            // Store the action space for state o
            // this is weird, it should be stored in a graph common to several occupancy states
            occupancy_state->setActionSpaceAt(t, joint_ddr_space);
        }
        // Return the action space corresponding to this ostate and t.
        return occupancy_state->getActionSpaceAt(t);
    }

    template <class TOccupancyState>
    std::shared_ptr<Action> BasePrivateOccupancyMDP<TOccupancyState>::getRandomAction(const std::shared_ptr<State> &ostate, number t)
    {
        if (this->store_actions_)
            return this->getActionSpaceAt(ostate, t)->sample()->toAction();
        else
            return this->computeRandomAction(ostate->toOccupancyState(), t);
    }

    template <class TOccupancyState>
    std::shared_ptr<Action> BasePrivateOccupancyMDP<TOccupancyState>::computeRandomAction(const std::shared_ptr<OccupancyStateInterface> &ostate, number t)
    {
        // Vector for storing individual decision rules.
        std::vector<std::shared_ptr<DecisionRule>> a;
        for (int agent = 0; agent < this->mdp->getNumAgents(); agent++)
        {
            // Input states for the a of agent.
            std::vector<std::shared_ptr<Item>> inputs;
            // Outputed actions for each of these.
            std::vector<std::shared_ptr<Item>> outputs;
            
            //TODO rename "OccupancyStateInterface::getIndividualHistories" by "OccupancyStateInterface::getIndividualDescriptiveStatistics"
            for (const auto &individual_descriptive_statistic : ostate->getIndividualHistories(agent))
            {
                inputs.push_back(individual_descriptive_statistic);
                outputs.push_back(this->decpomdp->getActionSpace(agent, t)->sample());
            }
            a.push_back(std::make_shared<DeterministicDecisionRule>(inputs, outputs));
        }
        return std::make_shared<JointDeterministicDecisionRule>(a, this->mdp->getActionSpace(t));
    }


    //Redefined to only return individual (pure) decision rules for the best-reponding player

        template <class TOccupancyState>
    std::shared_ptr<Space> BasePrivateOccupancyMDP<TOccupancyState>::computeActionSpaceAt(const std::shared_ptr<State> &ostate, number t)
    {
        // get the private histories of player num_player_ that is computing a BR.
        std::set<std::shared_ptr<HistoryInterface>> individual_descriptive_statistics = ostate->toOccupancyState()->getIndividualHistories(this->num_player_);
        
        // Get individual history space of agent i.
        std::shared_ptr<Space> individual_history_space = std::make_shared<DiscreteSpace>(sdm::tools::set2vector(individual_descriptive_statistics));
        
        // Get action space of agent i.
        std::shared_ptr<Space> individual_action_space = this->decpomdp->getActionSpace(this->num_player_, t);
        
        // Get individual ddr of agent i.
        std::shared_ptr<Space> individual_ddr_space = std::make_shared<FunctionSpace<DeterministicDecisionRule>>(individual_history_space, individual_action_space, this->store_actions_);
        
       return individual_ddr_space;
    }

    template <class TOccupancyState>
    std::shared_ptr<BasePrivateOccupancyMDP<TOccupancyState>> BasePrivateOccupancyMDP<TOccupancyState>::getptr()
    {
        return std::enable_shared_from_this<BasePrivateOccupancyMDP<TOccupancyState>>::shared_from_this();
    }

} // namespace sdm
