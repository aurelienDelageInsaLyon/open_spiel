#pragma once

#include <sdm/types.hpp>
#include <sdm/utils/struct/vector.hpp>
#include <sdm/utils/struct/pair.hpp>

namespace sdm
{
  class SerializedState : public Pair<number, std::vector<number>>
  {
  public:
    using state_type = number;
    using action_type = number;

    SerializedState();
    SerializedState(number state);
    SerializedState(number state, std::vector<number> actions);
    SerializedState(const SerializedState &v);

    /**
     * @brief Get the hidden state of the serializedState
     * 
     * @return number 
     */
    number getState() const;

    /**
     * @brief Get the hidden vector of action that were already decided
     * p
     * @return std::vector<number> 
     */
    std::vector<number> getAction() const;

    /**
     * @brief Get the current Agent Id of the object
     * 
     * @return number 
     */
    number getCurrentAgentId() const;

    /**
     *  @brief  Returns an ostream instance
     */
    friend std::ostream &operator<<(std::ostream &os, SerializedState &state)
    {
      os << "<serial-state agent_id=\"" << state.getCurrentAgentId() << "\" x=\"" << state.getState() << "\" actions=\"" << state.getAction() << "\"/>";
      return os;
    }
  };

} // namespace sdm

namespace boost
{
  namespace serialization
  {
    template <class Archive>
    void serialize(Archive &archive, sdm::SerializedState &serialized_state, const unsigned int)
    {
      archive &boost::serialization::base_object<sdm::Pair<sdm::number, std::vector<sdm::number>>>(serialized_state);
    }

  } // namespace serialization
} // namespace boost

namespace std
{
  template <>
  struct hash<sdm::SerializedState>
  {
    typedef sdm::SerializedState argument_type;
    typedef std::size_t result_type;
    inline result_type operator()(const argument_type &in) const
    {
      size_t seed = 0;
      //Combine the hash of the current vector with the hashes of the previous ones
      sdm::hash_combine(seed, in.first);
      sdm::hash_combine(seed, in.second);
      return seed;
    }
  };
}
