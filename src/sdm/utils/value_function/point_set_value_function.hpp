#pragma once
#define LOGTIME

#include <sdm/utils/value_function/initializer/initializer.hpp>
#include <sdm/utils/value_function/tabular_value_function.hpp>
#include <sdm/utils/value_function/backup/backup_interface.hpp>

/**
 * @brief Namespace grouping all tools required for sequential decision making.
 * @namespace  sdm
 */
namespace sdm
{
    template <class Hash = std::hash<std::shared_ptr<State>>, class KeyEqual = std::equal_to<std::shared_ptr<State>>>
    class BasePointSetValueFunction : public BaseTabularValueFunction<Hash, KeyEqual>
    {
    public:
        using Container = typename BaseTabularValueFunction<Hash, KeyEqual>::Container;

        BasePointSetValueFunction(number horizon, const std::shared_ptr<Initializer> &initializer, const std::shared_ptr<BackupInterfaceForValueFunction> &backup, const std::shared_ptr<ActionVFInterface> &action_vf, int freq_prunning = -1, TypeOfSawtoothPrunning type_of_sawtooth_prunning = TypeOfSawtoothPrunning::NONE);

        BasePointSetValueFunction(number horizon, double default_value = 0., const std::shared_ptr<BackupInterfaceForValueFunction> &backup = nullptr, const std::shared_ptr<ActionVFInterface> &action_vf = nullptr, int freq_prunning = -1, TypeOfSawtoothPrunning type_of_sawtooth_prunning = TypeOfSawtoothPrunning::NONE);

        std::string str() const;

        /**
         * @brief Evaluate the element given
         * 
         * @param state : ELement to evaluate
         * @param t 
         * @return Pair<std::shared_ptr<State>, double> 
         */
        Pair<std::shared_ptr<State>, double> evaluate(const std::shared_ptr<State> &state, number t);

        friend std::ostream &operator<<(std::ostream &os, BasePointSetValueFunction &vf)
        {
            os << vf.str();
            return os;
        }

        /**
         * @brief Evaluate the value at a state.
         * 
         * @param state the state where we want to evaluate the function
         * @return the value
         */
        double getValueAt(const std::shared_ptr<State> &state, number t);

        void do_pruning(number t);
        
    protected:

        bool is_sawtooth_lp = false;

        /**
         * @brief Frequency before prunning.
         */
        number freq_pruning_;

        /**
         * @brief The last time the prunning took place.
         * 
         */
        number last_prunning = 0;

        /**
         * @brief epsilon value used for the prunning
         * 
         */
        double epsilon_prunning;

        TypeOfSawtoothPrunning type_of_sawtooth_prunning_;

        /**
         * @brief Point-wise pruning.
         * 
         */
        void prune(number t = 0);

        /**
         * @brief COmpute the sawtooth ratio for the evaluate function 
         * 
         * @param state 
         * @param point 
         * @param t 
         * @return double 
         */
        double computeRatio(const std::shared_ptr<State> &state, const std::shared_ptr<State> &point, number t);

        /**
        * @brief Ratio specialized for the case Occupancy (used for the evaluate function)
        * 
        * @param state :Element to evaluate
        * @param point : Point in the Point Set
        * @return Pair<std::shared_ptr<State>,double> 
        */
        double ratioOccupancy(const std::shared_ptr<State> &state, const std::shared_ptr<State> &point,number t);

        /**
         * @brief Ratio specialized for the case Beleif (used for the evaluate function)
         * 
         * @param state : Element to evaluate
         * @param point :Point in the Point Set
         * @return Pair<std::shared_ptr<State>,double> 
         */
        double ratioBelief(const std::shared_ptr<State> &state, const std::shared_ptr<State> &point);

        Pair<std::unordered_map<std::shared_ptr<State>,std::vector<std::shared_ptr<State>>>,std::map<int,std::vector<std::shared_ptr<State>>>> iterative_pruning(number t);

    public:
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive &archive, const unsigned int)
        {
            using boost::serialization::make_nvp;

            archive &make_nvp("horizon", this->horizon_);
            archive &make_nvp("representation", this->representation);
        }
    };

    using PointSetValueFunction = BasePointSetValueFunction<std::hash<std::shared_ptr<State>>, std::equal_to<std::shared_ptr<State>>>;
    using PointSetValueFunction2 = BasePointSetValueFunction<sdm::hash_from_ptr<State>, sdm::equal_from_ptr<State>>;

} // namespace sdm

#include <sdm/utils/value_function/point_set_value_function.tpp>
