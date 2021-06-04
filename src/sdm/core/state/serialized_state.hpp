#pragma once

#include <sdm/types.hpp>
#include <sdm/core/state/base_state.hpp>
#include <sdm/core/action/action.hpp>

#include <sdm/utils/struct/vector.hpp>
#include <sdm/utils/struct/pair.hpp>

namespace sdm
{
  class SerializedState : public virtual BaseState<Pair<std::shared_ptr<State>, Joint<std::shared_ptr<Action>>>>  //public Pair<number, std::vector<number>>
  {
  public:
    SerializedState();
    SerializedState(std::shared_ptr<State> state, Joint<std::shared_ptr<Action>> actions);
    SerializedState(const SerializedState &v);

    std::shared_ptr<State> getHiddenState() const;

    /**
     * @brief Get the hidden vector of action that were already decided
     * p
     * @return std::vector<std::shared_ptr<Action>> 
     */
    Joint<std::shared_ptr<Action>> getAction() const;

    /**
     * @brief Get the current Agent Id of the object
     * 
     * @return number 
     */
    number getCurrentAgentId() const;

    /**
     * @brief Set the Agent Id of the object
     * 
     */
    void setAgentId(number);

    std::string str() const;

  protected : 

    number agentID_;
  };

} // namespace sdm

namespace boost
{
  namespace serialization
  {
    template <class Archive>
    void serialize(Archive &archive, sdm::SerializedState &serialized_state, const unsigned int)
    {
      // archive &boost::serialization::base_object<sdm::Pair<sdm::number, std::vector<sdm::number>>>(serialized_state);
    }

  } // namespace serialization
} // namespace boost


// namespace std
// {
//   template<>
//   struct hash<sdm::SerializedState>
//   {
//     typedef sdm::SerializedState argument_type;
//     typedef std::size_t result_type;
//     inline result_type operator()(const argument_type &in) const
//     {
//       size_t seed = 0;
//       //Combine the hash of the current vector with the hashes of the previous ones
//       sdm::hash_combine(seed, in.first);
//       sdm::hash_combine(seed, in.second);
//       return seed;
//     }
//   };
// }
