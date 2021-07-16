#include <sdm/utils/value_function/hyperplan_value_function.hpp>
#include <sdm/utils/value_function/backup/backup_base.hpp>
#include <sdm/core/state/interface/belief_interface.hpp>

#include <sdm/core/state/belief_state.hpp>
#include <sdm/core/state/belief_default.hpp>

#include <sdm/core/state/occupancy_state.hpp>

namespace sdm
{        

    double HyperplanValueFunction::PRECISION = config::PRECISION_SDMS_VECTOR;

    HyperplanValueFunction::HyperplanValueFunction(number horizon, const std::shared_ptr<Initializer> &initializer, const std::shared_ptr<BackupInterfaceForValueFunction> &backup, const std::shared_ptr<ActionVFInterface> &action_vf, int freq_prunning,TypeOfMaxPlanPrunning type_of_maxplan_prunning)
        : ValueFunction(horizon, initializer,backup,action_vf), freq_prune_(freq_prunning), type_of_maxplan_prunning_(type_of_maxplan_prunning)
    {
        this->representation = std::vector<HyperplanSet>(this->isInfiniteHorizon() ? 1 : this->horizon_ + 1, HyperplanSet({}));
        this->all_state_updated_so_far = std::vector<std::unordered_set<std::shared_ptr<State>>>(this->isInfiniteHorizon() ? 1 : this->horizon_ + 1,std::unordered_set<std::shared_ptr<State>>());
        this->default_values_per_horizon = std::vector<double>(this->isInfiniteHorizon() ? 1 : this->horizon_ + 1, 0);
    }

    HyperplanValueFunction::HyperplanValueFunction(number horizon,double default_value, const std::shared_ptr<BackupInterfaceForValueFunction> &backup, const std::shared_ptr<ActionVFInterface> &action_vf, int freq_prunning,TypeOfMaxPlanPrunning type_of_maxplan_prunning)
        : HyperplanValueFunction(horizon, std::make_shared<ValueInitializer>(default_value),backup,action_vf,freq_prunning,type_of_maxplan_prunning){}

    HyperplanValueFunction::~HyperplanValueFunction(){}

    void HyperplanValueFunction::initialize(double value, number t)
    {
        this->default_values_per_horizon[t] = value;
    }
    
    void HyperplanValueFunction::initialize()
    {
        this->initializer_->init(this->getptr());
    }

    double HyperplanValueFunction::getValueAt(const std::shared_ptr<State> &state, number t)
    {
        return this->evaluate(state, t).second;
    }

    void HyperplanValueFunction::updateValueAt(const std::shared_ptr<State> &state, number t)
    {
        
        //Determine the new hyperplan
        const auto &new_hyperplan = this->template backup<std::shared_ptr<State>>(state,this->getBestAction(state,t),t)->toBelief();

        // If the hyperplan doesn't exit, we add it to representation at t
        if (!this->exist(new_hyperplan,t))
        {
            this->representation[t].push_back(new_hyperplan);

            // Add state to all state update so far, only if the prunning used is Bounded
            if(this->type_of_maxplan_prunning_ == TypeOfMaxPlanPrunning::BOUNDED)
                this->all_state_updated_so_far[t].insert(state);
        }
    }

    
    std::vector<std::shared_ptr<State>> HyperplanValueFunction::getSupport(number t)
    {
        return this->representation[t];
    }

    double HyperplanValueFunction::getDefaultValue(number t)
    {
        return this->default_values_per_horizon[t];
    }

    void HyperplanValueFunction::do_prunning(number t)
    {
        if (this->last_prunning == this->freq_prune_)
        {
            // std::cout<<"Search for prunning"<<std::endl;
            for (number time = 0; time < this->getHorizon(); time++)
            {
                this->prune(time);
            }
            this->last_prunning = 0;
        }
        this->last_prunning++;
    }
    
    void HyperplanValueFunction::prune(number t)
    {
        switch (this->type_of_maxplan_prunning_)
        {
        case TypeOfMaxPlanPrunning::PAIRWISE :
            this->pairwise_prune(t);
            break;
        case TypeOfMaxPlanPrunning::BOUNDED :
            this->bounded_prune(t);
         
        default:
            break;
        }
    }

    void HyperplanValueFunction::pairwise_prune(number t)
    {
        std::vector<std::shared_ptr<BeliefInterface>> hyperplan_not_to_be_deleted;
        std::vector<std::shared_ptr<BeliefInterface>> hyperplan_to_delete;

        // Go over all hyperplan
        for (const auto &alpha : this->getSupport(t))
        {
            bool alpha_dominated = false;

            //Go over all hyperplan in hyperplan_not_to_be_deleted
            for (const auto &beta : hyperplan_not_to_be_deleted)
            {
                // If beta dominate alpha, we had alpha to the hyperplan to delete
                if (alpha->toBelief()->operator<(beta))
                {
                    hyperplan_to_delete.push_back(alpha->toBelief());
                    alpha_dominated = true;
                    break;
                }
            }
            // If alpha is dominated, we go to the next hyperplan
            if (alpha_dominated)
            {
                continue;
            }

            //Go over all hyperplan in hyperplan_not_to_be_deleted
            std::vector<std::shared_ptr<BeliefInterface>> erase_tempo;

            for(const auto &beta : hyperplan_not_to_be_deleted)
            {
                //If alpha dominate a vector in hyperplan_not_to_be_deleted, we deleted this vector
                if (beta->operator<(alpha->toBelief()))
                {
                    erase_tempo.push_back(beta);
                    hyperplan_to_delete.push_back(beta);
                }
            }

            for (const auto &erase : erase_tempo)
            {
                auto it = std::find(hyperplan_not_to_be_deleted.begin(), hyperplan_not_to_be_deleted.end(), erase);
                hyperplan_not_to_be_deleted.erase(it);
            }
            hyperplan_not_to_be_deleted.push_back(alpha->toBelief());
        }

        for(const auto &to_delete : hyperplan_to_delete)
        {
            this->representation[t].erase(std::find(this->representation[t].begin(), this->representation[t].end(), to_delete));
        }
    }
    
    void HyperplanValueFunction::bounded_prune(number t)
    {
        // Pour bounded prunning, il faut aussi noter les points intéressants , et chercher pour ces points la


        std::unordered_map<std::shared_ptr<State>, number> refCount;
        auto all_plan = this->getSupport(t);

        // std::cout<<"All hyperplan "<<std::endl;
        // Initialize ref count to 0 for each hyperplan
        // for (auto iter = all_plan.begin(); iter != all_plan.end(); iter++)
        // {
        //     // std::cout<<"Element "<<(*iter)->str()<<std::endl;
        //     refCount.emplace(*iter, 0);
        // }

        for(const auto&element : all_plan)
        {
            refCount[element] = 0;
        }

        //<! update the count
        std::shared_ptr<State> max_alpha;
        double max_value = -std::numeric_limits<double>::max(), value;
        for (const auto &hyperplan : this->all_state_updated_so_far[t])
        {
            // std::cout<<"Element in all state updated so far "<<hyperplan->str()<<std::endl;

            for (const auto &alpha : refCount)
            {
                // std::cout<<"Alpha "<<alpha.first->str()<<std::endl;
                // std::cout<<"Value "<< (hyperplan->toBelief()->operator^(alpha.first->toBelief()))<<std::endl;
                if (max_value < (value = (hyperplan->toBelief()->operator^(alpha.first->toBelief()))) )
                {
                    max_value = value;
                    max_alpha = alpha.first;
                }
            }
            refCount.at(max_alpha)++;
        }

        for (const auto& element : all_plan)
        {
            if (refCount.at(element) == 0)
            {
                // std::cout<<"Hyperplan to delete "<<(*iter)->str()<<std::endl;
                this->representation[t].erase(std::find(this->representation[t].begin(), this->representation[t].end(), element));
            }
        }
    }

    bool HyperplanValueFunction::exist(const std::shared_ptr<BeliefInterface>& new_vector,number t, double )
    {
        #ifdef LOGTIME
            this->StartTime();
        #endif

        // Go over all element in the Support
        for(const auto& element : this->representation[t])
        {
            // Test if the new vector is equal to the element 
            if(new_vector->operator==(element->toBelief()))
            {
                #ifdef LOGTIME
                    this->updateTime("Exist");
                #endif

                return true;
            } 
        }

        #ifdef LOGTIME
            this->updateTime("Exist");
        #endif

        return false;
    }

    Pair<std::shared_ptr<State>,double> HyperplanValueFunction::evaluate(const std::shared_ptr<State> &state, number t)
    {
        try
        {
            #ifdef LOGTIME
                this->StartTime();
            #endif

            double current, max = -std::numeric_limits<double>::max();
            std::shared_ptr<BeliefInterface> alpha_vector;

            auto belief_state = state->toBelief();

            //Create Default State
            this->createDefault(state,t);

            // Go over all hyperplan in the support
            for (const auto &plan : this->getSupport(t))
            {
                auto belief_plan = plan->toBelief();

                //Determine the best hyperplan which give the best value for the current state
                if (max < (current = belief_state->operator^(belief_plan) ))
                {
                    max = current;
                    alpha_vector = belief_plan;
                }
            }

            #ifdef LOGTIME
                this->updateTime("Evaluate");
            #endif
            
            return {alpha_vector,max};
        }
        catch(const std::exception& e)
        {
            std::cerr <<"HyperplanValueFunction::evaluate error"<< e.what() << '\n';
            exit(-1);
        }
    }

    void HyperplanValueFunction::createDefault(const std::shared_ptr<State>& state, number t)
    {
        // If there are not element at time t, we have to create the default State
        if(this->representation[t].size() == 0)
        {
            //Create the default state
            std::shared_ptr<BeliefInterface> default_state;

            switch (state->getTypeState())
            {
            case TypeState::BELIEF_STATE:
                default_state =  std::make_shared<Belief>();
                break;
            case TypeState::OCCUPANCY_STATE:
                default_state =  std::make_shared<OccupancyState>();
                break;
            default:
                throw sdm::exception::Exception("The initializer used is not available for this formalism !");
                break;
            }

            // Add default value of the default state
            default_state->setDefaultValue(this->getDefaultValue(t));
            this->representation[t].push_back(default_state);
            // default_state->finalize();
        }
    }

    std::string HyperplanValueFunction::str() const
    {
        std::ostringstream res;
        res << "<hyperplan_value_function horizon=\"" << ((this->isInfiniteHorizon()) ? "inf" : std::to_string(this->getHorizon())) << "\">" << std::endl;

        for (number i = 0; i < this->representation.size(); i++)
        {
            res << "\t<value timestep=\"" << ((this->isInfiniteHorizon()) ? "all" : std::to_string(i)) << ">" << std::endl;
            for (auto plan : this->representation[i])
            {
                res << "\t\t<plan>" << std::endl;
                res << "\t\t\t" << plan->str() << std::endl;
                res << "\t\t</plan>" << std::endl;
            }
            res << "\t</value>" << std::endl;
        }

        res << "</hyerplan_value_function>" << std::endl;
        return res.str();
    }

    size_t HyperplanValueFunction::getSize(number t) const
    {
        return this->representation[this->isInfiniteHorizon() ? 0 : t].size();
    }
} // namespace sdm