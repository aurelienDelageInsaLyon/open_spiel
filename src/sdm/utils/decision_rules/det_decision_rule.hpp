
#pragma once

#include <map>

#include <sdm/core/function.hpp>
#include <sdm/utils/linear_algebra/mapped_vector.hpp>
#include <sdm/tools.hpp>

/**
 * @brief Namespace grouping all tools required for sequential decision making.
 * @namespace  sdm
 */
namespace sdm
{

  /**
   * @brief The deterministic decision rule class. This class is a function that maps generic states to generic actions. 
   * 
   * @tparam TState the state type
   * @tparam TAction the action type
   */
  template <typename TState, typename TAction>
  class DeterministicDecisionRule : public Function<TState, TAction>
  {
  protected:
    std::map<TState, TAction> container_;

  public:
    DeterministicDecisionRule() {}
    
    DeterministicDecisionRule(std::vector<TState> acc_states, std::vector<TAction> n_actions)
    {
      assert(acc_states.size() == n_actions.size());
      for (int i = 0; i < acc_states.size(); i++)
      {
        this->container_[acc_states[i]] = n_actions[i];
      }
    }

    TAction act(const TState &s) const
    {
      return this->container_.at(s);
    }

    TAction operator()(const TState &s) const
    {
      return this->container_.at(s);
    }

    bool operator<(const DeterministicDecisionRule &v2) const
    {
      return this->container_ < v2.container_;
    }

    friend std::ostream &operator<<(std::ostream &os, const DeterministicDecisionRule<TState, TAction> &dr)
    {
      os << "<decision-rule type=\"deterministic\">" << std::endl;
      for (auto v : dr.container_)
      {
        os << "\t<decision state=\"" << v.first << "\">" << std::endl;
        std::ostringstream res;
        res << "\t\t" << v.second << std::endl;
        sdm::tools::indentedOutput(os, res.str().c_str());
        // os << "\t\t" << v.second << std::endl;
        os << "\t<decision/>" << std::endl;
      }
      os << "<decision-rule/>" << std::endl;
      return os;
    }
  };

  template <typename TState, typename TAction>
  using DetDecisionRule = DeterministicDecisionRule<TState, TAction>;
} // namespace sdm
