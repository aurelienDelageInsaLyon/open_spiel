#include <iomanip>
#include <sdm/config.hpp>
#include <sdm/exception.hpp>
#include <sdm/core/state/belief_state.hpp>

namespace sdm
{
  double Belief::PRECISION = config::PRECISION_BELIEF;

  Belief::Belief() : MappedVector<std::shared_ptr<State>>(0.)
  {
  }

  Belief::Belief(std::size_t size) : MappedVector<std::shared_ptr<State>>(size, 0.)
  {
  }

  Belief::Belief(const std::vector<std::shared_ptr<State>> &list_states, const std::vector<double> &list_proba) : Belief(list_states.size())
  {
    assert(list_states.size() == list_proba.size());
    for (size_t i = 0; i < list_states.size(); i++)
    {
      this->setProbability(list_states[i], list_proba[i]);
    }
  }

  Belief::Belief(const Belief &v) : MappedVector<std::shared_ptr<State>>(v), distribution_(v.distribution_)
  {
  }

  Belief::Belief(const MappedVector<std::shared_ptr<State>> &v) : MappedVector<std::shared_ptr<State>>(v)
  {
    this->finalize();
  }

  Belief::~Belief()
  {
  }

  std::vector<std::shared_ptr<State>> Belief::getStates() const
  {
    return this->getIndexes();
  }

  bool Belief::isStateExist(const std::shared_ptr<State> &state_tmp) const
  {
    return MappedVector<std::shared_ptr<State>, double>::isExist(state_tmp);
  }

  void Belief::setProbability(const std::shared_ptr<State> &state, double proba)
  {
    // Set the new occupancy measure
    this->setValueAt(state, proba);
  }

  double Belief::getProbability(const std::shared_ptr<State> &state) const
  {
    return this->getValueAt(state);
  }

  void Belief::addProbability(const std::shared_ptr<State> &state, double proba)
  {
    this->setValueAt(state, this->getProbability(state) + proba);
  }

  std::shared_ptr<State> Belief::sampleState()
  {
    return this->distribution_.sample();
  }

  void Belief::normalizeBelief(double norm_1)
  {
    if (norm_1 > 0)
    {
      for (const auto &state : this->getStates())
      {
        this->setProbability(state, this->getProbability(state) / norm_1);
      }
    }
  }

  std::shared_ptr<State> Belief::getState(const std::shared_ptr<State> &state)
  {
    return state;
  }

  size_t Belief::hash(double precision) const
  {
    if (precision < 0)
    {
      precision = Belief::PRECISION;
    }
    return std::hash<Belief>()(*this, precision);
  }
  
  bool Belief::operator==(const Belief &other) const
  {
    return this->isEqual(other);
  }

  bool Belief::isEqual(const Belief &other, double precision) const
  {
    if (precision < 0)
    {
      precision = Belief::PRECISION;
    }
    return MappedVector<std::shared_ptr<State>, double>::isEqual(other, precision);
  }

  bool Belief::isEqual(const std::shared_ptr<State> &other, double precision) const
  {
    return this->isEqual(*std::dynamic_pointer_cast<Belief>(other), precision);
  }

  bool Belief::operator==(const std::shared_ptr<BeliefInterface> &other) const
  {
    if (this->size() != other->size())
    {
      return false;
    }

    for (const auto &state : this->getStates())
    {
      if (this->getProbability(state) != other->getProbability(state))
      {
        return false;
      }
    }
    return true;
  }

  double Belief::operator^(const std::shared_ptr<BeliefInterface> &other) const
  {
    double product = 0.0;

    for (const auto &item : *this)
    {
      product += item.second * other->getProbability(item.first);
    }
    return product;
  }

  double Belief::operator<(const Belief &other) const
  {
    return MappedVector<std::shared_ptr<State>, double>::operator<(other);
  }

  double Belief::operator<(const std::shared_ptr<BeliefInterface> &other) const
  {
    return MappedVector<std::shared_ptr<State>, double>::operator<(*std::dynamic_pointer_cast<Belief>(other));
  }

  double Belief::norm_1() const
  {
    return MappedVector<std::shared_ptr<State>, double>::norm_1();
  }

  TypeState Belief::getTypeState() const
  {
    return TypeState::BELIEF_STATE;
  }

  void Belief::setDefaultValue(double default_value)
  {
    this->setDefault(default_value);
  }

  double Belief::getDefaultValue() const
  {
    return this->getDefault();
  }

  std::shared_ptr<BeliefInterface::Vector> Belief::getVectorInferface()
  {
    return std::dynamic_pointer_cast<MappedVector<std::shared_ptr<State>>>(this->getPointer());
  }

  void Belief::finalize()
  {
    MappedVector<std::shared_ptr<State>>::finalize();
    for (const auto &state : this->getStates())
    {
      this->distribution_.setProbability(state, this->getProbability(state));
    }
  }

  std::string Belief::str() const
  {
    std::ostringstream res;
    res << std::setprecision(config::BELIEF_DECIMAL_PRINT) << std::fixed;

    res << "BeliefState[" << MappedVector<std::shared_ptr<State>>::size() << "]( ";
    int i = 0;
    for (const auto &pair_state_proba : *this)
    {
      res << ((i == 0) ? "" : " | ");
      res << pair_state_proba.first->str() << " : " << pair_state_proba.second;
      i++;
    }
    res << ", default value =";
    res << this->getDefaultValue() << " )";
    return res.str();
  }

} // namespace sdm
