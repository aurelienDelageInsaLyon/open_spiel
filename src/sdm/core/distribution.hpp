/**
 * @file action.hpp
 * @author David Albert (david.albert@insa-lyon.fr)
 * @brief The file for interface action class
 * @version 0.1
 * @date 11/12/2020
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#pragma once

#include <sdm/types.hpp>
#include <sdm/core/item.hpp>

namespace sdm
{
    template <typename T>
    class Distribution
    {
    public:
        virtual T sample() const = 0;
        virtual double getProbability(const T &begin, const T &end) const = 0;
    };

    template <typename T>
    class DiscreteDistribution : public Distribution<T> 
    {
    public:
        virtual T sample() const;

        virtual double getProbability(const T &begin, const T & = 0) const;

        virtual double setProbability(const T &item, double proba) const;

    protected:
        std::map<T, double> probabilities_;
    };
} // namespace sdm
#include <sdm/core/distribution.tpp>