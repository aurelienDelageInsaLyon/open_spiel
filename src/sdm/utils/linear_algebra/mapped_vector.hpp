/**
 * @file mapped_vector.hpp
 * @author David Albert (david.albert@insa-lyon.fr)
 * @brief 
 * @version 0.1
 * @date 07/01/2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#pragma once

#include <unordered_map>
#include <cmath>
#include <string>
#include <vector>

#include <iostream>
#include <assert.h>

#include <sdm/types.hpp>
#include <sdm/utils/linear_algebra/vector_impl.hpp>
#include <sdm/utils/struct/pair.hpp>
#include <sdm/utils/struct/recursive_map.hpp>

namespace sdm
{
    /**
     * @brief Mapped vectors are vectors that use map to store values of a vector. 
     * 
     * Using map structure allow to store only necessary values. Moreover, it allows to see vector in another way that is not more only a mapping from integer to value but can map any type of index to values.   
     * 
     * @tparam TIndex Type of index
     * @tparam T Type of value
     */
    template <typename TIndex, typename T = double>
    class MappedVector : public RecursiveMap<TIndex, T>, public VectorImpl<TIndex, T>
    {
    protected:
        T default_value_;
        long size_ = -1;

        std::pair<TIndex, T> getMin() const;
        std::pair<TIndex, T> getMax() const;

    public:
        using iterator = typename std::unordered_map<TIndex, T>::iterator;
        using const_iterator = typename std::unordered_map<TIndex, T>::const_iterator;

        using type = typename RecursiveMap<TIndex, T>::type;
        using value_type = typename RecursiveMap<TIndex, T>::value_type;
        using value_list_type = typename RecursiveMap<TIndex, T>::value_list_type;

        MappedVector();
        MappedVector(T default_value);
        MappedVector(long size, T default_value);
        MappedVector(const MappedVector &v);
        MappedVector(std::initializer_list<value_list_type> vals) : RecursiveMap<TIndex, T>(vals) {}
        virtual ~MappedVector();

        T norm_1() const;
        T norm_2() const;

        T min() const;
        TIndex argmin() const;

        T max() const;
        TIndex argmax() const;

        T at(const TIndex &) const;
        T operator^(const MappedVector &) const;

        bool operator==(const MappedVector &v2) const;
        bool operator!=(const MappedVector &v2) const;
        bool operator<(const MappedVector &) const;

        T dot(const MappedVector &v2) const;

        std::size_t size() const;

        T getDefault() const;

        std::vector<TIndex> getIndexes() const;

        std::string str() const;

        friend std::ostream &operator<<(std::ostream &os, const MappedVector &vect)
        {
            os << vect.str();
            return os;
        }
    };
} // namespace sdm
#include <sdm/utils/linear_algebra/mapped_vector.tpp>