// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/matching_pennies.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace matching_pennies {
namespace {

constexpr int kNumPlayers = 2;
constexpr int kNumDistinctActions = 2;
constexpr int kNumStates = 1;
constexpr int kNumObservations = 1;


// Rewards.
constexpr double kWin = -1;
constexpr double kLoose = 2;

// Default parameters.
constexpr int kDefaultHorizon =2;
constexpr bool kDefaultFullyObservable = false;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"matching_pennies",
    /*long_name=*/"Matching Pennies",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {{"fully_observable", GameParameter(kDefaultFullyObservable)},
     {"horizon", GameParameter(kDefaultHorizon)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MatchingPenniesGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

ActionType ToAction(Action action) {
  switch (action) {
    case 0:
      return ActionType::kHead;
    case 1:
      return ActionType::kTail;
  }
  SpielFatalError(absl::StrCat("Invalid action: ", action));
}

std::string ActionToString(Action action) {
  switch (action) {
    case 0:
      return "play head";
    case 1:
      return "play tail";
  }

  SpielFatalError(absl::StrCat("Invalid action: ", action));
}

}  // namespace

MatchingPenniesState::MatchingPenniesState(std::shared_ptr<const Game> game,
                                         int horizon, bool fully_observable)
    : SimMoveState(game),
      total_rewards_(0),
      horizon_(horizon),
      cur_player_(kChancePlayerId),
      total_moves_(0),
      initiative_(0),
      win_(false),
      fully_observable_(fully_observable),
      reward_(0),
      lastActions_({0.0,0.0}),
      cur_state_(hiddenState::si)
      {}

std::string MatchingPenniesState::ActionToString(Player player, Action action) const {
  return std::to_string(action);
}
std::string MatchingPenniesState::stateToString(hiddenState hiddenState_) const {
    switch (hiddenState_) {
      case hiddenState::si:
        return "si";
      case hiddenState::sh:
        return "sh";
      case hiddenState::st:
      return "st";
    }
    return "";
  }

std::string MatchingPenniesState::InformationStateString(Player player) const {
  const MatchingPenniesGame& game = open_spiel::down_cast<const MatchingPenniesGame&>(*game_);
  //return MatchingPenniesState::History();
  std::vector<MatchingPenniesState::PlayerAction> hist = MatchingPenniesState::FullHistoryOfPlayer(player);
  std::vector<std::string> histString;
  std::transform(std::begin(hist),
               std::end(hist), 
               std::back_inserter(histString),
               [](MatchingPenniesState::PlayerAction d) { return std::to_string(d.action); } 
              );
  //std::cout << "\n \n i'm returning :  " << std::accumulate(histString.begin(), histString.end(), std::string(""));
  return std::accumulate(histString.begin(), histString.end(), std::string(""));
}

void MatchingPenniesState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), 2);
  SPIEL_CHECK_EQ(cur_player_, kSimultaneousPlayerId);
  /*
  moves_[0] = ToAction(actions[0]);
  moves_[1] = ToAction(actions[1]);
  cur_player_ = kChancePlayerId;*/
  std::cout << "simultaneous play \n";
  if (IsSimultaneousNode()){ 
    //std::cout << "doapplyactions : increasing timestep_";
    //std::cout << " timestep_ : " + std::to_string(timestep_);
    //ApplyActions(actions);
    //ApplyActions does push-back joint action to history
    //compute the rewards
    //std::cout << "\n computed rewards : " << MatchingPenniesState::resolveRewards(actions);
    reward_+= MatchingPenniesState::resolveRewards(actions);
    //std::cout << "\n rewards : " << reward_;
    cur_player_ = kChancePlayerId;
    std::cout << "\ntimestep_ : " << timestep_;
    std::cout << "\n players played : " << actions;
    timestep_++;
    //std::cout << "\nthe state should evolve from :";
    //std::cout << "\nprevious state : " << MatchingPenniesState::map_to_string(distribHidden_States);
    //MatchingPenniesState::evolveState(actions);
    //std::cout << "\nnew state : " << MatchingPenniesState::map_to_string(distribHidden_States) << "\n\n";
    //action_status_ = {ToAction(actions[0]), ToAction(actions[1])};
    lastActions_ = actions;
    //std::cout << "\nchanged last actions for : " << actions;
    total_moves_++;
  }
  else{
    //ApplyActions(actions);
    timestep_++;
    cur_player_ = kChancePlayerId;
    std::cout << "shouldn't happen";
    //shouldn't happen
    return;
  }
}

//need to add reward?
//histories are filled up by apply actions()
void MatchingPenniesState::DoApplyAction(Action action) {
  //std::cout << "doapplyaction : increasing timestep_";
  //std::cout << " timestep_ : " + std::to_string(timestep_);
  //timestep_++;
  //reward_ = 0;
  //ApplyAction(action);
  //std::cout << "chance plays\n";
  if (IsSimultaneousNode()){
    std::cout << "weird, shouldn't be a simultaneous node";
    return;
  }
  
  //chance node
  else{
    std::cout << "\n chance plays " << action;
    /*
    if (action == 0){
      distribHidden_States = {{hiddenState::si,1.0}, {hiddenState::st,0.0}, {hiddenState::sh,0.0}};
    }
    if (action == 1){
      distribHidden_States = {{hiddenState::si,0.0}, {hiddenState::st,1.0}, {hiddenState::sh,0.0}};
    }
    if (action == 2){
      distribHidden_States = {{hiddenState::si,0.0}, {hiddenState::st,0.0}, {hiddenState::sh,1.0}};
    }*/
    evolveState(lastActions_);
    cur_player_ = kSimultaneousPlayerId;
  }
}


void MatchingPenniesState::evolveState(const std::vector<Action>& actions){
  //encodes T(s,a,s') ? or will it be done by chance outcomes? to select (i) observations players

std::map<hiddenState,double> newDistribHidden_States;
  //need to initialize it;
  if (timestep_==0){//initial belief
    distribHidden_States = {{hiddenState::si,1.0}, {hiddenState::st,0.0}, {hiddenState::sh,0.0}};
    return;
  }
  std::cout<< "\n last action : " << lastActions_;
  for (std::pair<hiddenState,double> hiddenStatePair : distribHidden_States){
    hiddenState hiddenState_ = hiddenStatePair.first;
    if (ToAction(actions[0]) == ActionType::kHead &&
          ToAction(actions[1]) == ActionType::kHead ){
        std::cout<< "\nplayers play Head,Head";
        for (std::pair<hiddenState,double> newHiddenStatePair : distribHidden_States){
          hiddenState newHiddenState_ = newHiddenStatePair.first;
            if (newHiddenState_ == hiddenState::si){
              newDistribHidden_States[newHiddenState_] += 0*distribHidden_States.at(hiddenState_);
            }
            if (newHiddenState_ == hiddenState::st){
              newDistribHidden_States[newHiddenState_] += 1.0*distribHidden_States.at(hiddenState_);  
            }
            if (newHiddenState_ == hiddenState::sh){
              newDistribHidden_States[newHiddenState_] += 0.0*distribHidden_States.at(hiddenState_);
            }
          }
      }
      if (ToAction(actions[0]) == ActionType::kTail &&
          ToAction(actions[1]) == ActionType::kTail ){
                std::cout<< "\nplayers play Tail,Tail";
for (std::pair<hiddenState,double> newHiddenStatePair : distribHidden_States){
          hiddenState newHiddenState_ = newHiddenStatePair.first;
            if (newHiddenState_ == hiddenState::si){
              newDistribHidden_States[newHiddenState_] += 0*distribHidden_States.at(hiddenState_);
            }
            if (newHiddenState_ == hiddenState::st){
              newDistribHidden_States[newHiddenState_] += 0.0*distribHidden_States.at(hiddenState_);  
            }
            if (newHiddenState_ == hiddenState::sh){
              newDistribHidden_States[newHiddenState_] += 1.0*distribHidden_States.at(hiddenState_);
            }
          }
      }
      if (ToAction(actions[0]) == ActionType::kHead &&
          ToAction(actions[1]) == ActionType::kTail ){
                std::cout<< "\nplayers play Head,Tail";
for (std::pair<hiddenState,double> newHiddenStatePair : distribHidden_States){
          hiddenState newHiddenState_ = newHiddenStatePair.first;
            if (newHiddenState_ == hiddenState::si){
              newDistribHidden_States[newHiddenState_] += 0*distribHidden_States.at(hiddenState_);
            }
            if (newHiddenState_ == hiddenState::st){
              newDistribHidden_States[newHiddenState_] += 1.0*distribHidden_States.at(hiddenState_);  
            }
            if (newHiddenState_ == hiddenState::sh){
              newDistribHidden_States[newHiddenState_] += 0.0*distribHidden_States.at(hiddenState_);
            }
          }
      }
      if (ToAction(actions[0]) == ActionType::kTail &&
          ToAction(actions[1]) == ActionType::kHead ){
                std::cout<< "\nplayers play Tail,Head";
      for (std::pair<hiddenState,double> newHiddenStatePair : distribHidden_States){
          hiddenState newHiddenState_ = newHiddenStatePair.first;
            if (newHiddenState_ == hiddenState::si){
              newDistribHidden_States[newHiddenState_] += 0*distribHidden_States.at(hiddenState_);
            }
            if (newHiddenState_ == hiddenState::st){
              newDistribHidden_States[newHiddenState_] += 0.0*distribHidden_States.at(hiddenState_);  
            }
            if (newHiddenState_ == hiddenState::sh){
              newDistribHidden_States[newHiddenState_] += 1.0*distribHidden_States.at(hiddenState_);
            }
          }
      }
  }
  distribHidden_States = newDistribHidden_States;
  std::cout << "\n actions : " << actions << "evolves in : "  << distribHidden_States[hiddenState::si] << ", " << distribHidden_States[hiddenState::st] 
  << ", " << distribHidden_States[hiddenState::sh] << "\n" << std::flush;
  return;
}


double MatchingPenniesState::resolveRewards(const std::vector<Action>& actions){
  double sum = 0;
  std::cout << "\n for actions : " << actions << " in state  : "
   << distribHidden_States[hiddenState::si] << " , " 
   <<distribHidden_States[hiddenState::sh] << " , "
   << distribHidden_States[hiddenState::st];
  for (std::pair<hiddenState,double> hiddenStatePair : distribHidden_States){
    hiddenState hiddenState_ = hiddenStatePair.first;
    if (ToAction(actions[0]) == ActionType::kHead &&
          ToAction(actions[1]) == ActionType::kHead ){
            if (hiddenState_ == hiddenState::si){
              sum+= 0;
            }
            if (hiddenState_ == hiddenState::st){
              sum += -1*distribHidden_States.at(hiddenState_);
            }
            if (hiddenState_ == hiddenState::sh){
              sum += 2*distribHidden_States.at(hiddenState_);
            }
      }
      if (ToAction(actions[0]) == ActionType::kTail &&
          ToAction(actions[1]) == ActionType::kTail ){
            if (hiddenState_ == hiddenState::si){
              sum += 0;

            }
            if (hiddenState_ == hiddenState::st){
              sum += 1*distribHidden_States.at(hiddenState_);
            }
            if (hiddenState_ == hiddenState::sh){
              sum += -1*distribHidden_States.at(hiddenState_);
            }
      }
      if (ToAction(actions[0]) == ActionType::kHead &&
          ToAction(actions[1]) == ActionType::kTail ){
            if (hiddenState_ == hiddenState::si){
              sum += 0;
            }
            if (hiddenState_ == hiddenState::st){
              sum += 1*distribHidden_States.at(hiddenState_);
            }
            if (hiddenState_ == hiddenState::sh){
              sum += -1*distribHidden_States.at(hiddenState_);
            }
      }
      if (ToAction(actions[0]) == ActionType::kTail &&
          ToAction(actions[1]) == ActionType::kHead ){
            if (hiddenState_ == hiddenState::si){
              sum += 0;
            }
            if (hiddenState_ == hiddenState::st){
              sum += -1*distribHidden_States.at(hiddenState_);
            }
            if (hiddenState_ == hiddenState::sh){
              sum += +2*distribHidden_States.at(hiddenState_);
            }
      }
    }
    std::cout << "sum of rewards : " << sum;
    return sum;
}

std::vector<Action> MatchingPenniesState::LegalActions(Player player) const {
  //std::cout << "\n player : " << player;
  //std::cout << "\n cur_player_ : " << cur_player_;
  //std::cout << "\n ksimultaneous : " << kSimultaneousPlayerId;
  if (player == 0 || player == 1) {
    return {0,1};
  } else if (IsTerminal()) {
    return {};
  } else //chance node
    return {0};
}
std::string MatchingPenniesState::map_to_string(std::map<hiddenState,double>  &m) const {
  std::string output = "";
  std::string convrt = "";
  std::string result = "";

  for (auto it = m.cbegin(); it != m.cend(); it++) {
    convrt = std::to_string(it->second);
    output += MatchingPenniesState::stateToString(it->first) + ":" + (convrt) + ", ";
  }
  result = output.substr(0, output.size() - 2 );
  return result;
}

hiddenState actionToState(Action action){
  switch (action) {
    case 0:
      return hiddenState::si;
    case 1:
      return hiddenState::sh;
    case 2:
      return hiddenState::st;
  }
}

ActionsAndProbs MatchingPenniesState::ChanceOutcomes() const {
    SPIEL_CHECK_TRUE(IsChanceNode());
  
  std::map<hiddenState,double> newDistribHidden_States;
  //need to initialize it;

  const std::vector<Action> & actions = lastActions_;
  
  //std::cout << "\nhChanceOutcomesChanceOutcomesistory_ : " << history_;
  //std::cout << "\nlastActions_ : " << lastActions_;
  //std::cout << "\ncurrent distrib : " << map_to_string(distribHidden_States);
  
  //return {{0,1.0},{1,0.0},{2,0.0}};

  return {{0,1.0}};
  /*
  if (timestep_==0){
    return {{0,1.0},{1,0.2},{2,0.8}};
  }

  for (std::pair<hiddenState,double> hiddenStatePair : distribHidden_States){
    //hiddenState hiddenState_ = actionToState(history_.back().action);
    hiddenState hiddenState_ = hiddenStatePair.first;
    if (ToAction(actions[0]) == ActionType::kHead &&
          ToAction(actions[1]) == ActionType::kHead ){
        //std::cout<< "\nplayers play Head,Head";
        for (std::pair<hiddenState,double> newHiddenStatePair : distribHidden_States){
          hiddenState newHiddenState_ = newHiddenStatePair.first;
            if (newHiddenState_ == hiddenState::si){
              newDistribHidden_States[newHiddenState_] += 0*distribHidden_States.at(hiddenState_);
            }
            if (newHiddenState_ == hiddenState::st){
              newDistribHidden_States[newHiddenState_] += 0.0*distribHidden_States.at(hiddenState_);  
            }
            if (newHiddenState_ == hiddenState::sh){
              newDistribHidden_States[newHiddenState_] += 1.0*distribHidden_States.at(hiddenState_);
            }
          }
      }
      if (ToAction(actions[0]) == ActionType::kTail &&
          ToAction(actions[1]) == ActionType::kTail ){
        //std::cout<< "\nplayers play Tail,Tail";
for (std::pair<hiddenState,double> newHiddenStatePair : distribHidden_States){
          hiddenState newHiddenState_ = newHiddenStatePair.first;
            if (newHiddenState_ == hiddenState::si){
              newDistribHidden_States[newHiddenState_] += 0*distribHidden_States.at(hiddenState_);
            }
            if (newHiddenState_ == hiddenState::st){
              newDistribHidden_States[newHiddenState_] += 1.0*distribHidden_States.at(hiddenState_);  
            }
            if (newHiddenState_ == hiddenState::sh){
              newDistribHidden_States[newHiddenState_] += 0.0*distribHidden_States.at(hiddenState_);
            }
          }
      }
      if (ToAction(actions[0]) == ActionType::kHead &&
          ToAction(actions[1]) == ActionType::kTail ){
        //std::cout<< "\nplayers play Head,Tail";
for (std::pair<hiddenState,double> newHiddenStatePair : distribHidden_States){
          hiddenState newHiddenState_ = newHiddenStatePair.first;
            if (newHiddenState_ == hiddenState::si){
              newDistribHidden_States[newHiddenState_] += 0*distribHidden_States.at(hiddenState_);
            }
            if (newHiddenState_ == hiddenState::st){
              newDistribHidden_States[newHiddenState_] += 0.0*distribHidden_States.at(hiddenState_);  
            }
            if (newHiddenState_ == hiddenState::sh){
              newDistribHidden_States[newHiddenState_] += 1.0*distribHidden_States.at(hiddenState_);
            }
          }
      }
      if (ToAction(actions[0]) == ActionType::kTail &&
          ToAction(actions[1]) == ActionType::kHead ){
        //std::cout<< "\nplayers play Tail,Head";
for (std::pair<hiddenState,double> newHiddenStatePair : distribHidden_States){
          hiddenState newHiddenState_ = newHiddenStatePair.first;
            if (newHiddenState_ == hiddenState::si){
              newDistribHidden_States[newHiddenState_] += 0*distribHidden_States.at(hiddenState_);
            }
            if (newHiddenState_ == hiddenState::st){
              newDistribHidden_States[newHiddenState_] += 1.0*distribHidden_States.at(hiddenState_);  
            }
            if (newHiddenState_ == hiddenState::sh){
              newDistribHidden_States[newHiddenState_] += 0.0*distribHidden_States.at(hiddenState_);
            }
          }
      }
    }
  //}
  //distribHidden_States = newDistribHidden_States;
  //std::cout << "\nreturning distrib : " << map_to_string(newDistribHidden_States);

  return {{0,newDistribHidden_States[hiddenState::si]},{1,newDistribHidden_States[hiddenState::sh]},{2,newDistribHidden_States[hiddenState::st]}};
  
  */
}

std::string MatchingPenniesState::ToString() const {
  std::string result = "";
  absl::StrAppend(&result, "Total moves: ", total_moves_, "\n");
  absl::StrAppend(&result, "Most recent reward: ", reward_, "\n");
  absl::StrAppend(&result, "Total rewards: ", total_rewards_, "\n");

  return result;
}

/*
std::string MatchingPenniesState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (fully_observable_) {
    return ToString();
  } else {
    ObservationType obs = PartialObservation(player);
    switch (obs) {
      case kEmptyFieldObs:
        return "field";
      case kWallObs:
        return "wall";
      case kOtherAgentObs:
        return "other agent";
      case kSmallBoxObs:
        return "small box";
      case kBigBoxObs:
        return "big box";
      default:
        SpielFatalError("Unrecognized observation!");
    }
  }
}*/


bool MatchingPenniesState::IsTerminal() const { 
  //return winner_ != kInvalidPlayer; 
  //std::cout << "is terminal ? timestep_ : " << std::to_string(timestep_);
  if (horizon_==timestep_){
    return true;
  }
  return false;
}

std::vector<double> MatchingPenniesState::Returns() const {
  // Cooperative game: all players get same reward.
  return {reward_, -reward_};
}

std::vector<double> MatchingPenniesState::Rewards() const {
  // Cooperative game: all players get same reward.
  return {reward_, -reward_};
}
/*
void MatchingPenniesState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (fully_observable_) {
    TensorView<3> view(values, {kCellStates, kRows, kCols}, true);

    for (int r = 0; r < kRows; r++) {
      for (int c = 0; c < kCols; c++) {
        int plane = ObservationPlane({r, c}, player);
        SPIEL_CHECK_TRUE(plane >= 0 && plane < kCellStates);
        view[{plane, r, c}] = 1.0;
      }
    }
  } else {
    SPIEL_CHECK_EQ(values.size(), kNumObservations);
    std::fill(values.begin(), values.end(), 0);
    ObservationType obs = PartialObservation(player);
    values[obs] = 1;
  }
}*/

std::unique_ptr<State> MatchingPenniesState::Clone() const {
  return std::unique_ptr<State>(new MatchingPenniesState(*this));
}

MatchingPenniesGame::MatchingPenniesGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      horizon_(ParameterValue<int>("horizon")),
      fully_observable_(ParameterValue<bool>("fully_observable")) {}

std::vector<int> MatchingPenniesGame::ObservationTensorShape() const {
  if (fully_observable_) {
    return {3};
  } else {
    return {kNumObservations};
  }
}

int MatchingPenniesGame::NumDistinctActions() const {
  return kNumDistinctActions;
}

int MatchingPenniesGame::NumPlayers() const { return kNumPlayers; }

std::unique_ptr<State> MatchingPenniesGame::NewInitialState() const {
  std::unique_ptr<State> state(
      new MatchingPenniesState(shared_from_this(), horizon_, fully_observable_));
  return state;
}

double MatchingPenniesGame::MaxUtility() const {
  return (MaxGameLength() *  kLoose);
}

double MatchingPenniesGame::MinUtility() const {
  return (MaxGameLength() * kWin);
}

}  // namespace MP
}  // namespace open_spiel
