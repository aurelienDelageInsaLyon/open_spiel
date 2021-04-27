/**
 * @file max_plan_vf.hpp
 * @author David Albert (david.albert@insa-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 18/12/2020
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#pragma once
#include <set>

#include <sdm/core/state/serialized_state.hpp>
#include <sdm/utils/linear_algebra/vector.hpp>
#include <sdm/world/solvable_by_hsvi.hpp>
#include <sdm/utils/value_function/initializer.hpp>
#include <sdm/utils/value_function/value_function.hpp>

/**
 * @brief Namespace grouping all tools required for sequential decision making.
 * @namespace  sdm
 */
namespace sdm
{
    /**
     * @brief 
     * 
     * @tparam TVector type of hyperplan representation. Must implement sdm::VectorImpl interface.
     * @tparam TValue value type (default : double)
     */
    template <typename TVector, typename TAction, typename TValue = double>
    class MaxPlanValueFunction : public ValueFunction<TVector, TAction, TValue>
    {
    protected:
        using HyperplanSet = std::vector<TVector>;

        /**
         * @brief The value function represention.
         * The default representation is a MappedVector but every class implementing VectorImpl interface can be used.
         */
        std::vector<HyperplanSet> representation;

        /**
         * @brief The initializer to use for this value function. 
         * 
         */
        std::shared_ptr<Initializer<TVector, TAction>> initializer_;

        // std::vector<TValue> default_value_;

    public:
        MaxPlanValueFunction();
        MaxPlanValueFunction(std::shared_ptr<SolvableByHSVI<TVector, TAction>>, number, std::shared_ptr<Initializer<TVector, TAction>>);
        MaxPlanValueFunction(std::shared_ptr<SolvableByHSVI<TVector, TAction>>, number = 0, TValue = 0.);

        void initialize();
        void initialize(TValue, number = 0);

        /**
         * @brief Get the Value at state x 
         * 
         * @param state the state 
         * @return TValue 
         */
        TValue getValueAt(const TVector &, number = 0);

        /**
         * @brief Update the max plan representation by adding a new hyperplan
         */
        void updateValueAt(const TVector &, number = 0);

        /**
         * @brief 
         * 
         * @return std::string 
         */
        std::vector<TVector> getSupport(number);

        /**
         * @brief Get the maximum value and hyperplan at a specific state
         * 
         * @param state a specific state
         * @return the maximum value and hyperplan at a specific state (std::pair<TValue, TVector>) 
         */
        std::pair<TValue, TVector> getMaxAt(const TVector &, number);

        /**
         * @brief Prune unecessary vectors
         * 
         */
        void prune(number = 0);

        /*!
         * @brief this method prunes dominated alpha-vectors, known as Lark's pruning.
         * This approach goes other all vectors until all of them have been treated. For each arbitrary vector,
         * it performs a linear programming providing a gap delta, and a frequency f. If the gap is over a certain
         * threshold epsilon, that means one can preserve the selected vector, otherwise one should discard it.
         */
        void lark_prune(number = 0);

        /*!
         * @brief this method prunes dominated points, known as bounded pruning by Trey Smith.
         * This approach stores the number of frequency states, among those already visited, that are maximal at a hyperplan.
         * And prune hyperplan with a number of maximal frequency states zero.
         */
        void bounded_prune(number = 0);

        /**
         * @brief Get the number of hyperplans 
         * 
         * @return number 
         */
        number size();

        std::string str()
        {
            std::ostringstream res;
            res << "<maxplan_value_function horizon=\"" << ((this->isInfiniteHorizon()) ? "inf" : std::to_string(this->getHorizon())) << "\">" << std::endl;
            
            for (number i = 0; i < this->representation.size(); i++)
            {
                res << "\t<value timestep=\"" << ((this->isInfiniteHorizon()) ? "all" : std::to_string(i)) << ">" << std::endl;
                for (auto plan : this->representation[i])
                {
                    res << "\t\t<plan>" << std::endl;
                    res << "\t\t\t" << plan << std::endl;
                    res << "\t\t</plan>" << std::endl;
                }
                res << "\t</value>" << std::endl;
            }

            res << "</maxplan_value_function>" << std::endl;
            return res.str();
        }

        /**
         * @warning je ne comprends pas ce qui est fait ci-dessous. C'est etrange ...  pour moi c'est contraire a ce qu'on attend des templates -- ici on duplique les choses ... etrange vraiment etrange ...
         */

        // For POMDP (i.e. BeliefState as vector type)
        template <typename T, std::enable_if_t<std::is_same_v<BeliefState, T>, int> = 0>
        TVector backup_operator(const TVector &, number = 0);

        // For DecPOMDP (i.e. OccupancyState as vector type)
        template <typename T, std::enable_if_t<std::is_same_v<OccupancyState<>, T>, int> = 0>
        TVector backup_operator(const TVector &, number = 0);

        // For SerializedDecPOMDP (i.e. SerializedOccupancyState as vector type)
        template <typename T, std::enable_if_t<std::is_same_v<SerializedOccupancyState<>, T>, int> = 0>
        TVector backup_operator(const TVector &, number = 0);
    };

    template <class TAction, class TValue>
    class MaxPlanValueFunction<number, TAction, TValue> : public ValueFunction<number, TAction, TValue>
    {
    public:
        MaxPlanValueFunction(std::shared_ptr<SolvableByHSVI<number, TAction>>, number, std::shared_ptr<Initializer<number, TAction>>)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = number.");
        }
        MaxPlanValueFunction()
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = number.");
        }

        void initialize()
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = number.");
        }
        void initialize(TValue, number = 0)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = number.");
        }

        TValue getValueAt(const number &, number = 0)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = number.");
        }

        void updateValueAt(const number &, number = 0)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = number.");
        }

        std::vector<number> getSupport(number)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = number.");
        }

        std::string str()
        {
            return "MaxPlanVF";
        }
    };

    template <class TAction, class TValue>
    class MaxPlanValueFunction<SerializedState, TAction, TValue> : public ValueFunction<SerializedState, TAction, TValue>
    {
    public:
        MaxPlanValueFunction(std::shared_ptr<SolvableByHSVI<SerializedState, TAction>>, number, std::shared_ptr<Initializer<SerializedState, TAction>>)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = SerializedState.");
        }

        MaxPlanValueFunction()
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = SerializedState.");
        }

        void initialize()
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = SerializedState.");
        }
        void initialize(TValue, number = 0)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = SerializedState.");
        }

        TValue getValueAt(const SerializedState &, number = 0)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = SerializedState.");
        }

        void updateValueAt(const SerializedState &, number = 0)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = SerializedState.");
        }

        std::vector<SerializedState> getSupport(number)
        {
            throw sdm::exception::Exception("MaxPlanVF cannot be used for State = SerializedState.");
        }

        std::string str()
        {
            return "MaxPlanVF";
        }
    };

} // namespace sdm
#include <sdm/utils/value_function/max_plan_vf.tpp>
