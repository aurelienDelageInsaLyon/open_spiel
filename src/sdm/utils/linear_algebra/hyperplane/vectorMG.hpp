#pragma once

#include <sdm/macros.hpp>
#include <sdm/exception.hpp>
#include <sdm/utils/linear_algebra/hyperplane/hyperplane.hpp>
#include <sdm/world/base/pomdp_interface.hpp>
#include <sdm/core/state/occupancy_state_mg.hpp>
#include <sdm/core/state/private_br_occupancy_state.hpp>

namespace sdm
{
    class VectorMG : public Hyperplane
    {
    public:
        static double PRECISION;

        VectorMG(std::map<std::shared_ptr<HistoryInterface>, double> & values, double default_value, const std::shared_ptr<OccupancyStateMG> & state_support, int agent_id_);
        VectorMG(double default_value, std::shared_ptr<OccupancyStateMG> & occ_state_support) : Hyperplane(default_value) {
            this->occ_support = occ_state_support;
        }
        
        bool isDominated(const Hyperplane &other) const;

        double getInitValue();
        
         double getValueAt(const std::shared_ptr<State> &x, const std::shared_ptr<HistoryInterface> &o) const ;
         void setValueAt(const std::shared_ptr<State> &x, const std::shared_ptr<HistoryInterface> &o, double value);

         double getValueAt(const std::shared_ptr<State> &x, const std::shared_ptr<HistoryInterface> &o, const std::shared_ptr<Action> &u) const;
         void setValueAt(const std::shared_ptr<State> &x, const std::shared_ptr<HistoryInterface> &o, const std::shared_ptr<Action> &u, double value);

         double getValueAt(const std::shared_ptr<HistoryInterface> &o) const;
         void setValueAt(const std::shared_ptr<HistoryInterface> &o, double value);

        size_t hash(double precision = -1) const;
        bool isEqual(const VectorMG &other, double precision=-1) const;
        bool isEqual(const std::shared_ptr<Hyperplane> &other, double precision=-1) const;

        std::string str() const;

        size_t size() const;

        std::shared_ptr<PrivateOccupancyState> getSigmaC(std::shared_ptr<HistoryInterface> history);
                    std::shared_ptr<OccupancyStateMG> occ_support;


        std::map<std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>>, double> treeDeltaStrategies;
        
    protected : 
            std::unordered_map<std::shared_ptr<HistoryInterface>, double> repr;
            int agent_id_;

    };
}

DEFINE_STD_HASH(sdm::VectorMG, sdm::VectorMG::PRECISION);
