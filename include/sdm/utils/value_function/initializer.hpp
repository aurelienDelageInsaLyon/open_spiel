#pragma once
#include <math.h>

#include <sdm/utils/value_function/value_function.hpp>

namespace sdm
{
    template <typename TState, typename TAction>
    class Initializer
    {
    public:
        virtual void init(ValueFunction<TState, TAction> *vf) = 0;
    };

    template <typename TState, typename TAction>
    class ValueInitializer : public Initializer<TState, TAction>
    {
    protected:
        double value;

    public:
        ValueInitializer(double v) : value(v)
        {
        }

        void init(ValueFunction<TState, TAction> *vf)
        {
            if (vf->getHorizon() < 1)
            {
                vf->initialize(this->value);
            }
            else
            {
                for (int t = 0; t < vf->getHorizon(); t++)
                {
                    vf->initialize(this->value, t);
                }
            }
        }
    };

    template <typename TState, typename TAction>
    class ZeroInitializer : public ValueInitializer<TState, TAction>
    {
    public:
        ZeroInitializer() : ValueInitializer<TState, TAction>(0)
        {
        }
    };

    template <typename TState, typename TAction>
    class BoundInitializer : public Initializer<TState, TAction>
    {
        double value_, discount_;

    public:
        BoundInitializer(double value, double discount) : value_(value), discount_(discount)
        {
        }

        void init(ValueFunction<TState, TAction> *vf)
        {
            if (vf->isInfiniteHorizon())
            {
                assert(this->discount_ < 1);
                double value;
                double factor = 0, comp = 0;
                int n = 0;
                do
                {
                    comp = factor;
                    factor += std::pow(this->discount_, n);
                    n++;
                } while ((factor - comp) > 0.0001);
                value = floor(this->value_ * factor) + 1;
                vf->initialize(value);
            }
            else
            {
                for (int t = 0; t < vf->getHorizon(); t++)
                {
                    vf->initialize(this->value_ * (vf->getHorizon() - t), t);
                }
            }
        }
    };

    template <typename TState, typename TAction>
    class MinInitializer : public BoundInitializer<TState, TAction>
    {
    public:
        MinInitializer(double min_reward, double discount) : BoundInitializer<TState, TAction>(min_reward, discount)
        {
        }
    };

    template <typename TState, typename TAction>
    class MaxInitializer : public BoundInitializer<TState, TAction>
    {
    public:
        MaxInitializer(double max_reward, double discount) : BoundInitializer<TState, TAction>(max_reward, discount)
        {
        }
    };

    // template <typename TState, typename TAction>
    // class MDPInitializer : public Initializer<TState, TAction>
    // {
    // protected:
    //     POMDP problem_;
    //     double discount_;

    // public:
    //     MDPInitializer(POMDP problem, double discount) : problem_(problem), discount_(discount)
    //     {
    //     }

    //     void init(ValueFunction<TState, TAction> *vf)
    //     {
    //         auto algo = sdm::algo::make("mapped_hsvi", this->problem->toMDP());
    //         algo->do_solve();
    //         algo->getLowerBound();
    //     }
    // };
} // namespace sdm
