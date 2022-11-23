#include <memory>
#include <sdm/world/occupancy_mg.hpp>
#include <sdm/world/registry.hpp>

namespace sdm
{


    template <class TOccupancyState>
    BaseOccupancyMG<TOccupancyState>::BaseOccupancyMG()
    {
    }

    template <class TOccupancyState>
    BaseOccupancyMG<TOccupancyState>::BaseOccupancyMG(Config config)
        : BaseOccupancyMDP<TOccupancyState>(config)
    {

    }

    template <class TOccupancyState>
    BaseOccupancyMG<TOccupancyState>::BaseOccupancyMG(const std::shared_ptr<MPOMDPInterface> &decpomdp, const number num_player_ , Config config)
        : BaseOccupancyMDP<TOccupancyState>(decpomdp,num_player_,config)
    {

    }
        template <class TOccupancyState>
    BaseOccupancyMG<TOccupancyState>::BaseOccupancyMG(const std::shared_ptr<MPOMDPInterface> &decpomdp, const number num_player_, int memory, bool store_states, bool store_actions, int batch_size)
    {
        this->store_states_ = store_states;
        this->store_actions_ = store_actions;
        this->batch_size_ = batch_size;
        this->mdp = decpomdp;

        // Initialize underlying belief mdp
        this->belief_mdp_ = std::make_shared<BeliefMDP>(decpomdp, batch_size, store_states, store_actions);

        // Initialize initial history
        this->initial_history_ = std::make_shared<JointHistoryTree>(this->mdp->getNumAgents(), -1);

        // Initialize initial occupancy state
        //TODO modifier pour rendre compatible avec factored_occupancy_state, i.e., en argument on trouve le modele decpomdp ou ndpomdp plus peut-etre la memoire.
        this->initial_state_ = (std::shared_ptr<OccupancyStateMG>) std::make_shared<OccupancyStateMG>(this->mdp->getNumAgents(), 0);

        this->initial_state_->toOccupancyState()->setProbability(this->initial_history_->toJointHistory(), this->belief_mdp_->getInitialState()->toBelief(), 1);
        this->initial_state_->toOccupancyState()->finalize();

        // Initialize Transition Graph
        this->mdp_graph_ = std::make_shared<Graph<std::shared_ptr<State>, Pair<std::shared_ptr<Action>, std::shared_ptr<Observation>>>>();
        this->mdp_graph_->addNode(this->initial_state_);

        this->reward_graph_ = std::make_shared<Graph<double, Pair<std::shared_ptr<State>, std::shared_ptr<Action>>>>();
        this->reward_graph_->addNode(0.0);
    }
    
    template <class TOccupancyState>
    std::shared_ptr<Space> BaseOccupancyMG<TOccupancyState>::getObservationSpaceAt(const std::shared_ptr<State> &, const std::shared_ptr<Action> &, number)
    {
        //std::cout << "\n returning no_obs" << std::flush;
        return std::make_shared<DiscreteSpace>(std::vector({sdm::NO_OBSERVATION}));
    }
    
    template <class TOccupancyState>
    std::vector<Action> BaseOccupancyMG<TOccupancyState>::getBaseActions(number agent_id_){
        return this->mdp->getActionSpaceAt(this->initial_state_,0);
    }

    template <class TOccupancyState>
    BaseOccupancyMG<TOccupancyState>::~BaseOccupancyMG()
    {
    }
    
     template <class TOccupancyState>
    std::shared_ptr<State> BaseOccupancyMG<TOccupancyState>::getInitialState()
    {
        //std::cout << "i'm casting the initial state in the getter";
        return (std::shared_ptr<OccupancyStateMG>) this->initial_state_;
    
    }
}