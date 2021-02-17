#include <sdm/utils/value_function/value_function.hpp>

namespace sdm
{
    template <typename TState, typename TAction, typename TValue>
    ValueFunction<TState, TAction, TValue>::ValueFunction(std::shared_ptr<SolvableByHSVI<TState, TAction>> problem, int horizon) : problem_(problem), horizon_(horizon)
    {
    }

    template <typename TState, typename TAction, typename TValue>
    TValue ValueFunction<TState, TAction, TValue>::operator()(const TState &state)
    {
        return this->getValueAt(state);
    }

    template <typename TState, typename TAction, typename TValue>
    std::shared_ptr<VectorImpl<TAction, TValue>> ValueFunction<TState, TAction, TValue>::getQValueAt(TState &state, int t)
    {
        std::shared_ptr<MappedVector<TAction, TValue>> q_s = std::make_shared<MappedVector<TAction, TValue>>();
        for (auto &a : this->getWorld()->getActionSpace(state).getAll())
        {
            (*q_s)[a] = this->getQValueAt(state, a, t);
        }
        return q_s;
    }

    template <typename TState, typename TAction, typename TValue>
    TValue ValueFunction<TState, TAction, TValue>::getQValueAt(TState &state, TAction &action, int t)
    {
        // implement bellman operator
        return this->getWorld()->getReward(state, action) + this->getWorld()->getDiscount() * this->getWorld()->getExpectedNextValue(this, state, action, t);
    }

    template <typename TState, typename TAction, typename TValue>
    TAction ValueFunction<TState, TAction, TValue>::getBestAction(TState &state, int t)
    {
        auto qvalues = this->getQValueAt(state, t);
        return qvalues->argmax();
    }

    template <typename TState, typename TAction, typename TValue>
    std::shared_ptr<SolvableByHSVI<TState, TAction>> ValueFunction<TState, TAction, TValue>::getWorld()
    {
        return this->problem_;
    }

    template <typename TState, typename TAction, typename TValue>
    int ValueFunction<TState, TAction, TValue>::getHorizon() const
    {
        return this->horizon_;
    }

    template <typename TState, typename TAction, typename TValue>
    bool ValueFunction<TState, TAction, TValue>::isFiniteHorizon() const
    {
        return (this->horizon_ > 0);
    }

    template <typename TState, typename TAction, typename TValue>
    bool ValueFunction<TState, TAction, TValue>::isInfiniteHorizon() const
    {
        return !(this->isFiniteHorizon());
    }
} // namespace sdm