#include <sdm/utils/value_function/vfunction/partial_Q_value_function.hpp>
#include <sdm/utils/linear_algebra/hyperplane/balpha.hpp>
#include <sdm/utils/linear_algebra/hyperplane/oalpha.hpp>

#include <sdm/core/state/interface/belief_interface.hpp>
#include <sdm/core/state/belief_state.hpp>
#include <sdm/core/state/belief_default.hpp>

#include <sdm/core/state/occupancy_state.hpp>
#include <sdm/core/state/serial_occupancy_state.hpp>

#include <sdm/utils/value_function/initializer/initializer.hpp>
#include <sdm/world/base/belief_mdp_interface.hpp>

#include <sdm/core/state/occupancy_state_mg.hpp>

namespace sdm
{

    double partialQValueFunction::PRECISION = config::PRECISION_SDMS_VECTOR;

    partialQValueFunction::partialQValueFunction(const std::shared_ptr<SolvableByDP> &world,
                                         const std::shared_ptr<Initializer> &initializer,
                                         const std::shared_ptr<ActionSelectionInterface> &action_selection,
                                         int agent_id_,
                                         int freq_pruning,
                                         MaxplanPruning::Type type_of_maxplan_prunning)
        : ValueFunctionInterface(world, initializer, action_selection),
          ValueFunction(world, initializer, action_selection_),
          type_of_maxplan_prunning_(type_of_maxplan_prunning),
          agent_id_(agent_id_)
    {
        // Create all different structure in order to use the hyperplan value function.
        this->representation = std::vector<HyperplanSet>(this->isInfiniteHorizon() ? 1 : world->getHorizon() + 1, HyperplanSet({}));
        this->all_state_updated_so_far = std::vector<std::unordered_set<std::shared_ptr<OccupancyStateMG>>>(this->isInfiniteHorizon() ? 1 : world->getHorizon() + 1, std::unordered_set<std::shared_ptr<OccupancyStateMG>>());
        this->default_values_per_horizon = std::vector<double>(this->isInfiniteHorizon() ? 1 : world->getHorizon() + 1, 0);
        this->pomdp = std::dynamic_pointer_cast<POMDPInterface>(world->getUnderlyingProblem());
    }

    partialQValueFunction::partialQValueFunction(const std::shared_ptr<SolvableByDP> &world,
                                         const std::shared_ptr<Initializer> &initializer,
                                         const std::shared_ptr<ActionSelectionInterface> &action_selection,
                                         Config config, int agent_id_)
        : partialQValueFunction(world, initializer, action_selection, config.get("freq_pruning", -1))
    {
        this->agent_id_ = agent_id_;
        auto opt_int = config.getOpt<int>("pruning_type");
        auto opt_str = config.getOpt<std::string>("pruning_type");
        if (opt_int.has_value())
        {
            this->type_of_maxplan_prunning_ = (MaxplanPruning::Type)opt_int.value();
        }
        else if (opt_str.has_value())
        {
            auto iter = MaxplanPruning::TYPE_MAP.find(opt_str.value());
            this->type_of_maxplan_prunning_ = (iter != MaxplanPruning::TYPE_MAP.end()) ? iter->second : MaxplanPruning::PAIRWISE;
        }
    }

    partialQValueFunction::partialQValueFunction(const partialQValueFunction &copy)
        : ValueFunctionInterface(copy.world_, copy.initializer_, copy.action_selection_),
          ValueFunction(copy.world_, copy.initializer_, copy.action_selection_),
          representation(copy.representation),
          default_values_per_horizon(copy.default_values_per_horizon),
          type_of_maxplan_prunning_(copy.type_of_maxplan_prunning_),
          all_state_updated_so_far(copy.all_state_updated_so_far),
          agent_id_(copy.agent_id_)
    {
    }

    std::vector<std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>> partialQValueFunction::getListWTuples(number t){
        return this->representationW[t];
    }



    std::shared_ptr<VectorMG> partialQValueFunction::getSupportVector(int t)
    {
        if (t==0){
            double max = -100000.0;
            int argmax = 0;
            for (int k = 0;k<this->representationV[0].size();k++){
                if (this->representationV[0][k].second->getInitValue()>max){
                    max = this->representationV[0][k].second->getInitValue();
                    argmax = k;
                }
            }
            std::cout << "\n value argmax : " << max;
            return this->representationV[0][argmax].second;
        }
         if (t==1){
            double max = 100000.0;
            int argmax = 0;
            for (int k = 0;k<this->representationV[0].size();k++){
                if (this->representationV[0][k].second->getInitValue()<max){
                    max = this->representationV[0][k].second->getInitValue();
                    argmax = k;
                }
            }
            std::cout << "\n value argmax : " << max;
            return this->representationV[0][argmax].second;
        }
        //std::cout << "\n blob, size : " << this->representationV[0].size();
        //return this->representationV[0][this->representationV[0].size()-1].second;
    }

    double partialQValueFunction::getValueAt(const std::shared_ptr<State> &state, number t){
        
        //std::cout << "\n here";
        
        double min = -(this->agent_id_*2-1)*world_->getUnderlyingProblem()->getMaxReward()*world_->getHorizon();
        double lambda = -(this->agent_id_*2-1) *world_->getUnderlyingProblem()->getMaxReward()*world_->getHorizon() - world_->getUnderlyingProblem()->getMinReward()*world_->getHorizon();
        //upb -> agent 0 maximizing -> \upbV = \min_{v} ...


        //std::shared_ptr<VectorMG> alpha_vector = nullptr;
        std::shared_ptr<VectorMG> vectorMG = nullptr;
        // Go over all hyperplan in the support
        std::shared_ptr<OccupancyStateMG> occupancy_state = std::dynamic_pointer_cast<OccupancyStateMG>(state);
        //std::cout << " \n t : " << t;
        

        for (const auto &plan : this->representationV[t])
        {
            //std::cout << "\n plan : " << plan.second->str();

            //TODO: change this to make a scalar product
            double valScalarProduct = 0.0;
            //std::cout << "\n starting the scalar product";
            for (auto & history : occupancy_state->getIndividualHistories(this->agent_id_)){
                valScalarProduct+= plan.second->getValueAt(history)*occupancy_state->getProbabilityOverIndividualHistories(this->agent_id_,history);
            }
            //std::cout << "\n value scal product : " << valScalarProduct;
                
            valScalarProduct+=  lambda * occupancy_state->norm_1_other(plan.first);
            
            if (valScalarProduct<min && this->agent_id_==0){
                min = valScalarProduct;
                //vectorMG = plan;    
            }
            if (valScalarProduct>min && this->agent_id_==1){
                min = valScalarProduct;
                //vectorMG = plan;    
            }
            
        }

        if (t==0){
        //std::cout << "\n returning min : " << min << " for player " << this->agent_id_;
        }
        return min;
    }


    void partialQValueFunction::initialize(double value, number t)
    {
        
    //std::cout << "initial state : " << this->getWorld()->getInitialState()->str();

        this->default_values_per_horizon[this->isInfiniteHorizon() ? 0 : t] = value;
        auto initial_state = std::dynamic_pointer_cast<OccupancyStateMG>(this->getWorld()->getInitialState());

        // If there are not element at time t, we have to create the default State
        if (this->representation[this->isInfiniteHorizon() ? 0 : t].size() == 0)
        {
            // Create the default state
            std::shared_ptr<VectorMG> default_hyperplane;
            if (sdm::isInstanceOf<OccupancyStateMG>(initial_state))
            {
                default_hyperplane = std::make_shared<VectorMG>(value,initial_state);
            }
            /*
            else if (sdm::isInstanceOf<BeliefInterface>(initial_state))
            {
                // default_hyperplane = std::make_shared<BeliefVectorMG>();
                default_hyperplane = std::make_shared<bAlpha>(value);
            }*/
            else
            {
                throw sdm::exception::TypeError("TypeError : state must derived from belief");
            }

            this->addHyperplaneAt(initial_state, default_hyperplane, t);
        }
    }

    void partialQValueFunction::initialize()
    {
        this->initializer_->init(this->getptr());
    }

    double partialQValueFunction::getValueAt(const std::shared_ptr<OccupancyStateMG> &state, number t)
    {
        std::cout << "\n should never be called for now ";
        std::exit(1);
        return this->evaluate(state, t).second;
    }
    

    std::map<std::shared_ptr<HistoryInterface>, double> partialQValueFunction::decompressValues(const std::shared_ptr<OccupancyStateMG> & state, std::map<std::shared_ptr<HistoryInterface>, double> values){
        std::map<std::shared_ptr<HistoryInterface>, double> uncompressed_values;
        for ( std::map<std::shared_ptr<HistoryInterface>, double>::const_iterator it = values.begin();it != values.end(); ++it){
            //uncompressed_values.emplace(it->first,it->second);
            //std::cout << "\n hist in compressed hist : " << it->first->str() << " value : " << it->second;
        }
        for (const std::shared_ptr<HistoryInterface> & hist : state->getOneStepUncompressedOccupancy()->getIndividualHistories(this->agent_id_)){
            //std::cout << "\n player : " << this->agent_id_ << " hist : " << hist->str();
            if (values.count(hist)==0){
                //std::cout << "\n should add a compressed hist value";
                //std::exit(1);
            }
            uncompressed_values.emplace(hist, values[state->getLabel(hist, this->agent_id_)]);
        }
        
        /*
        for ( std::map<std::shared_ptr<HistoryInterface>, double>::const_iterator it = values.begin();it != values.end(); ++it){
            uncompressed_values.emplace(it->first,it->second);
        }*/

        return uncompressed_values;
    }

    void partialQValueFunction::addTuple(const std::shared_ptr<OccupancyStateMG> & state, const std::shared_ptr<OccupancyStateMG> & lastState, const std::shared_ptr<StochasticDecisionRule> & lastStrat, std::map<std::shared_ptr<HistoryInterface>, double> values, std::map<std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>, double> treeDeltaStrategy, number t)
    {
        //std::cout << "\n adding tuple, t : " << t << " state : " << state->str();
        //std::cout << "\n value : " << values.begin()->second;
        //std::map<int,std::vector<std::tuple<std::shared_ptr<OccupancyStateMG>,StochasticDecisionRule,VectorMG>>> representationW;
        std::map<std::shared_ptr<HistoryInterface>, double> uncompressed_values = this->decompressValues(state,values);
        double def_value;
        if (this->agent_id_==0){
            def_value = this->pomdp->getMaxReward()*(this->getWorld()->getHorizon()-t);
        }
        else{
            def_value = this->pomdp->getMinReward()*(this->getWorld()->getHorizon()-t);

        }
        if (this->representationW.count(t)==0){
            
            std::vector<std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>> vec;
            std::shared_ptr<VectorMG> new_vector = std::make_shared<VectorMG>(uncompressed_values,def_value,state, this->agent_id_);
            vec.push_back(std::make_tuple(lastState,lastStrat,new_vector));

            new_vector->treeDeltaStrategies = treeDeltaStrategy;

            this->representationW.emplace(t,vec);

            std::vector<std::pair<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<VectorMG>>> vecForV;
            std::shared_ptr<VectorMG> new_vector_v = std::make_shared<VectorMG>(uncompressed_values,def_value,state, this->agent_id_);

            new_vector_v ->treeDeltaStrategies = treeDeltaStrategy;

            vecForV.push_back(std::make_pair(state,new_vector_v));

            this->representationV[t].push_back(std::make_pair(state,new_vector_v));
            //std::cout << "\n create timestep and added the vector : " << new_vector->str() << " \n at timestep : " << t;
        }
        else{
            std::shared_ptr<VectorMG> new_vector = std::make_shared<VectorMG>(uncompressed_values,def_value,state, this->agent_id_);

            new_vector ->treeDeltaStrategies = treeDeltaStrategy;

            this->representationW[t].push_back(std::make_tuple(lastState,lastStrat,new_vector));
            std::shared_ptr<VectorMG> new_vector_v = std::make_shared<VectorMG>(uncompressed_values,def_value,state, this->agent_id_);

            new_vector_v ->treeDeltaStrategies = treeDeltaStrategy;

            this->representationV[t].push_back(std::make_pair(state,new_vector_v));
            if (t==1){
                //std::cout << "\n player : " << this->agent_id_ << " : added the vector : " << new_vector->str() << " \n at timestep : " << t;
            }

        }
        if (t==0){
        //std::cout << "\n size of representation W : " << this->representationW[t].size();
        //std::cout << "\n size of representation V : " << this->representationV[t].size();
        }

    }

    Pair<std::shared_ptr<Hyperplane>, double> partialQValueFunction::evaluate(const std::shared_ptr<OccupancyStateMG> &state, number t)
    {
        std::cout << "\n should never be called";
        std::exit(1);
        double min = std::numeric_limits<double>::max();
        //std::shared_ptr<VectorMG> alpha_vector = nullptr;
        std::shared_ptr<VectorMG> vectorMG = nullptr;
        // Go over all hyperplan in the support
        for (const auto &plan : this->representationV[t])
        {
            //std::cout << "\n plan : " << plan;
            // Determine the best hyperplan which give the best value for the current state
            /*
            if (max < (current = state->product(plan)))
            {
                max = current;
                alpha_vector = plan;
            }*/

            //TODO: change this to make a scalar product
            /*
            double valScalarProduct = 0.0;
            for (auto & history : state->getIndividualHistories(this->agent_id_)){
                valScalarProduct+= plan.second->getValueAt(history)*state->getProbabilityOverIndividualHistories(this->agent_id_,history);
            }
            if (t==0){
                //std::cout << "\n value scalarproduct : " << valScalarProduct;
            }
            if (valScalarProduct<min){
                min = valScalarProduct;
                vectorMG = std::make_shared<VectorMG>(*plan.second);    
            }
*/
        }
        if (t==0){
        //std::cout << "\n min : " << min;
        }
        return {vectorMG, min};
    }

    void partialQValueFunction::addHyperplaneAt(const std::shared_ptr<OccupancyStateMG> &state, const std::shared_ptr<Hyperplane> &new_hyperplan, number t)
    {
        // Add hyperplane in the hyperplane set
        this->representation[this->isInfiniteHorizon() ? 0 : t].insert(std::static_pointer_cast<VectorMG>(new_hyperplan));

        // Add state to all state update so far, only if the prunning used is Bounded
        if (this->type_of_maxplan_prunning_ == MaxplanPruning::Type::BOUNDED)
            this->all_state_updated_so_far[this->isInfiniteHorizon() ? 0 : t].insert(state);
    }

    double partialQValueFunction::getBeta(const std::shared_ptr<Hyperplane> &hyperplane, const std::shared_ptr<OccupancyStateMG> &x, const std::shared_ptr<HistoryInterface> &o, const std::shared_ptr<Action> &u, number t)
    {   
        std::cout << "\n getBeta in partial_q_value_function should never be called, i'm exiting";
        std::exit(1);
        return 0.0;
        /*
        auto alpha = std::static_pointer_cast<VectorMG>(hyperplane);
        return alpha->getBetaValueAt(x, o, u, this->pomdp, t);*/
    }

    std::vector<std::shared_ptr<Hyperplane>> partialQValueFunction::getHyperplanesAt(std::shared_ptr<OccupancyStateMG>, number t)
    {
        auto set = this->representation[this->isInfiniteHorizon() ? 0 : t];
        return std::vector<std::shared_ptr<Hyperplane>>(set.begin(), set.end());
    }

    partialQValueFunction::HyperplanSet &partialQValueFunction::getAlphaHyperplanesAt(number t)
    {
        return this->representation[this->isInfiniteHorizon() ? 0 : t];
    }

    std::shared_ptr<Hyperplane> partialQValueFunction::getHyperplaneAt(std::shared_ptr<OccupancyStateMG> state, number t)
    {
        return this->evaluate(state, t).first;
    }

    std::vector<std::shared_ptr<State>> partialQValueFunction::getSupport(number t)
    {
        throw sdm::exception::NotImplementedException("NotImplementedException raised in partialQValueFunction::getSupport");
        // return this->getHyperplanesAt(nullptr, t);
    }

    std::shared_ptr<ValueFunctionInterface> partialQValueFunction::copy()
    {
        auto casted_value = std::dynamic_pointer_cast<partialQValueFunction>(this->getptr());
        return std::make_shared<partialQValueFunction>(*casted_value);
    }

    double partialQValueFunction::getDefaultValue(number t)
    {
        return this->default_values_per_horizon[this->isInfiniteHorizon() ? 0 : t];
    }

    void partialQValueFunction::prune(number t)
    {
        switch (this->type_of_maxplan_prunning_)
        {
        case MaxplanPruning::PAIRWISE:
            this->pairwise_prune(t);
            break;
        case MaxplanPruning::BOUNDED:
            this->bounded_prune(t);
        default:
            break;
        }
    }

    void partialQValueFunction::pairwise_prune(number t)
    {
        // List of hyperplanes that are \eps-dominated
        std::vector<std::shared_ptr<VectorMG>> hyperplan_to_delete;
        // List of hyperplanes that are not dominated
        std::vector<std::shared_ptr<VectorMG>> hyperplan_to_keep;

        auto &all_hyperplanes = this->getAlphaHyperplanesAt(t);

        // Go over all current hyperplanes
        for (const auto &alpha : all_hyperplanes)
        {
            bool alpha_dominated = false;

            // Go over all hyperplanes in hyperplan_to_keep
            for (auto beta_iter = hyperplan_to_keep.begin(); beta_iter != hyperplan_to_keep.end(); beta_iter++)
            {
                // If beta dominate alpha, we add alpha to the set of hyperplanes to delete
                if (alpha->isDominated(**beta_iter))
                {
                    hyperplan_to_delete.push_back(alpha);
                    alpha_dominated = true;
                    break;
                }
            }
            // If alpha is dominated, we reject alpha and go to the next iteration of outer loop
            if (alpha_dominated)
                continue;

            // Go over all hyperplanes in hyperplan_to_keep
            for (auto beta_iter = hyperplan_to_keep.begin(); beta_iter != hyperplan_to_keep.end();)
            {
                // If alpha dominate a vector in hyperplan_to_keep, we deleted this vector
                if (alpha->dominate(**beta_iter))
                {
                    hyperplan_to_delete.push_back(*beta_iter);
                    beta_iter = hyperplan_to_keep.erase(beta_iter);
                }
                else
                {
                    beta_iter++;
                }
            }

            hyperplan_to_keep.push_back(alpha);
        }

        // Erase dominated hyperplanes
        for (const auto &to_delete : hyperplan_to_delete)
        {
            all_hyperplanes.erase(std::find(all_hyperplanes.begin(), all_hyperplanes.end(), to_delete));
        }
    }

    void partialQValueFunction::bounded_prune(number t)
    {
        /*
        std::unordered_map<std::shared_ptr<Hyperplane>, number> refCount;
        auto &all_hyperplanes = this->getAlphaHyperplanesAt(t);

        // Initialize the count
        for (const auto &hyperplane : all_hyperplanes)
        {
            refCount[hyperplane] = 0;
        }

        // Update the count depending on visited beliefs
        std::shared_ptr<Hyperplane> max_alpha;
        double max_value = -std::numeric_limits<double>::max(), value;
        for (const auto &hyperplane : this->all_state_updated_so_far[t])
        {
            for (const auto &alpha : all_hyperplanes)
            {
                if (max_value < (value = (hyperplane->product(alpha))))
                {
                    max_value = value;
                    max_alpha = alpha;
                }
            }
            refCount.at(max_alpha)++;
        }

        // Delete hyperplanes with a count of 0
        for (auto hyperplane_iter = all_hyperplanes.begin(); hyperplane_iter != all_hyperplanes.end();)
        {
            if (refCount.at(*hyperplane_iter) == 0)
                hyperplane_iter = all_hyperplanes.erase(hyperplane_iter);
            else
                hyperplane_iter++;
        }
        */
    }

    std::string partialQValueFunction::str() const
    {
        std::ostringstream res;
        res << "<pwlc_value_function horizon=\"" << ((this->isInfiniteHorizon()) ? "inf" : std::to_string(this->getHorizon())) << "\">" << std::endl;

        for (number i = 0; i < this->representation.size(); i++)
        {
            res << "\t<value timestep=\"" << ((this->isInfiniteHorizon()) ? "all" : std::to_string(i)) << ">" << std::endl;
            for (const auto &plan : this->representation[i])
            {
                std::ostringstream hyperplan_str;
                hyperplan_str << plan->str();
                tools::indentedOutput(res, hyperplan_str.str().c_str(), 2);
            }
            res << "\t</value>" << std::endl;
        }

        res << "</pwlc_value_function>" << std::endl;
        return res.str();
    }

    size_t partialQValueFunction::getSize(number t) const
    {
        return this->representation[this->isInfiniteHorizon() ? 0 : t].size();
    }




} // namespace sdm