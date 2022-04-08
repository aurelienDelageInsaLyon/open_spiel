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

#include "open_spiel/games/mabc.h"

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
namespace mabc {
namespace {

constexpr int kNumPlayers = 2;
constexpr int kNumDistinctActions = 2;
constexpr int kNumStates = 4;
constexpr int kNumObservations = 4;


// Rewards.
constexpr double kRMax = 0.4;
constexpr double kRMin = 0.0;

// Default parameters.
constexpr int kDefaultHorizon = 2;
constexpr bool kDefaultFullyObservable = false;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"mabc",
    /*long_name=*/"MABC",
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
  return std::shared_ptr<const Game>(new MABCGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

ActionType ToAction(Action action) {
  switch (action) {
    case 0:
      return ActionType::kSend;
    case 1:
      return ActionType::kWait;
  }
  SpielFatalError(absl::StrCat("Invalid action: ", action));
}

std::string ActionToString(Action action) {
  switch (action) {
    case 0:
      return "00";
    case 1:
      return "01";
  }

  SpielFatalError(absl::StrCat("Invalid action: ", action));
}

hiddenState actionToHiddenState(Action action){
  switch (action) {
    case 0:
      return hiddenState::s00;
    case 1:
      return hiddenState::s01;
    case 2:
      return hiddenState::s10;
    case 3:
      return hiddenState::s11;
  }
}

std::string observationToString(Action z) {
  switch (z) {
    case 00:
      return "Collision";
    case 01:
      return "No-Collisions";
    case 10:
      return "Collision";
    case 11:
      return "No-Collisions";
  }

  SpielFatalError(absl::StrCat("Invalid observation: ", z));
}


}  // namespace

MABCState::MABCState(std::shared_ptr<const Game> game,
                                         int horizon, bool fully_observable)
    : SimMoveState(game),
      total_rewards_(0),
      horizon_(horizon),
      cur_player_(kSimultaneousPlayerId),
      total_moves_(0),
      initiative_(0),
      win_(false),
      fully_observable_(fully_observable),
      reward_(0),
      lastActions_({0,0}),
      cur_state_(hiddenState::s11),
      distribHidden_States({{hiddenState::s00,0.0}, {hiddenState::s01,0.0}, {hiddenState::s10,0.0},{hiddenState::s11,1.0}})
      {}


std::string MABCState::ActionToString(Player player, Action action) const {
  return std::to_string(action);
}
std::string MABCState::observationToString(Action action,Player player) const {
  if (player==0){
    switch (action) {
      case 00:
        return ("-CollisionPlayer0-");
      case 01:
        return ("-CollisionPlayer0-");
      case 10:
        return ("-No-CollisionPlayer0-");
      case 11:
        return ("-No-CollisionPlayer0-");
    }
  }
  else{
    switch (action) {
      case 00:
        return ("-CollisionPlayer1-");
      case 01:
        return ("-No-CollisionPlayer1-");
      case 10:
        return ("-CollisionPlayer1-");
      case 11:
        return ("-No-CollisionPlayer1-");
    }
  }
}
std::string MABCState::actionToString(Action action,Player player) const {
  switch (action) {
    case 0:
      return ("a0p"+std::to_string(player));
    case 1:
      return ("a1p"+std::to_string(player));
  }
}
std::string MABCState::stateToString(hiddenState hiddenState_) const {
    switch (hiddenState_) {
      case hiddenState::s00:
        return "s00";
      case hiddenState::s01:
        return "s01";
      case hiddenState::s10:
        return "s10";
      case hiddenState::s11:
        return "s11";
    }
    return "";
  }

std::string MABCState::InformationStateString(Player player) const {
  const MABCGame& game = open_spiel::down_cast<const MABCGame&>(*game_);
  //return MABCState::History();
  std::vector<MABCState::PlayerAction> hist = MABCState::FullPrivateHistory(player);
  std::vector<std::string> histString;
  std::transform(std::begin(hist),
               std::end(hist), 
               std::back_inserter(histString),
               [this,&player](MABCState::PlayerAction d) { if (d.player==-1){ return observationToString(d.action,player);} else {return actionToString(d.action,d.player);} } 
              );
  //std::cout << "\n \n i'm returning :  " << std::accumulate(histString.begin(), histString.end(), std::string(""));
  return std::accumulate(histString.begin(), histString.end(), std::string(""));
}

void MABCState::DoApplyActions(const std::vector<Action>& actions) {
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
    //std::cout << "\n computed rewards : " << MABCState::resolveRewards(actions);
    reward_+= MABCState::resolveRewards(actions);
    //std::cout << "\n rewards : " << reward_;
    cur_player_ = kChancePlayerId;
    //std::cout << "\ntimestep_ : " << timestep_;
    std::cout << "\n players played : " << actions << " at timestep " << timestep_;
    timestep_++;
    //std::cout << "\nthe state should evolve from :";
    //std::cout << "\nprevious state : " << MABCState::map_to_string(distribHidden_States);
    //MABCState::evolveState(actions);
    //std::cout << "\nnew state : " << MABCState::map_to_string(distribHidden_States) << "\n\n";
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
void MABCState::DoApplyAction(Action action) {
  if (IsSimultaneousNode()){
    std::cout << "weird, shouldn't be a simultaneous node";
    return;
  }
  
  //chance node
  else{
    std::cout << "\n chance plays " << action << " at timestep : " << timestep_;
    //evolveState(lastActions_,action);
    //cur_player_ = kSimultaneousPlayerId;
    
    if (rollout_chance_==1){
      std::cout << "\n rolling out";

      rollout_action_=action;
      rollout_chance_=0; // reinitializes
      cur_player_ = kSimultaneousPlayerId;

    }
    else{
      std::cout << "\n rolling in";
      cur_state_ = actionToHiddenState(action);
      rollout_chance_=1;
    }
  }
}


std::map<hiddenState,double> MABCState::evolveState(const std::vector<Action>& actions, const Action actionNature) const{

  std::map<hiddenState,double> newDistribHidden_States;
  //need to initialize it;
  if (timestep_==0){//initial belief
    return {{hiddenState::s00,0.0}, {hiddenState::s01,0.0}, {hiddenState::s10,0.0}, {hiddenState::s11,1.0}};
  }
  //std::cout<< "\n last action : " << lastActions_;
  
  //for (std::pair<hiddenState,double> hiddenStatePair : distribHidden_States){
    hiddenState hiddenState_ = actionToHiddenState(actionNature);
    //send,send
    if (ToAction(actions[0]) == ActionType::kSend &&
          ToAction(actions[1]) == ActionType::kSend ){
      //hiddenState newHiddenState_ = newHiddenStatePair.first;
      if (hiddenState_ == hiddenState::s00){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s01){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s10){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s11){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
    }

    //send,wait
    if (ToAction(actions[0]) == ActionType::kSend &&
          ToAction(actions[1]) == ActionType::kWait ){
      if (hiddenState_ == hiddenState::s00){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s01){
        newDistribHidden_States[hiddenState::s00] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.1*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.9*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s10){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s11){
        newDistribHidden_States[hiddenState::s00] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.1*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.9*distribHidden_States.at(hiddenState_);
      }
    }

    //wait,send
    if (ToAction(actions[0]) == ActionType::kWait &&
          ToAction(actions[1]) == ActionType::kSend ){
      if (hiddenState_ == hiddenState::s00){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s01){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s10){
        newDistribHidden_States[hiddenState::s00] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.9*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.1*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s11){
        newDistribHidden_States[hiddenState::s00] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.9*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.1*distribHidden_States.at(hiddenState_);
      }
    }

    //wait,wait
    if (ToAction(actions[0]) == ActionType::kWait &&
          ToAction(actions[1]) == ActionType::kWait ){
      if (hiddenState_ == hiddenState::s00){
        newDistribHidden_States[hiddenState::s00] += 0.09*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.01*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.81*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.09*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s01){
        newDistribHidden_States[hiddenState::s00] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.1*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.9*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s10){
        newDistribHidden_States[hiddenState::s00] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.9*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 0.1*distribHidden_States.at(hiddenState_);
      }
      if (hiddenState_ == hiddenState::s11){
        newDistribHidden_States[hiddenState::s00] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s01] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s10] += 0.0*distribHidden_States.at(hiddenState_);
        newDistribHidden_States[hiddenState::s11] += 1.0*distribHidden_States.at(hiddenState_);
      }
    }

  //distribHidden_States = newDistribHidden_States;
  /*
  for (std::pair<hiddenState,double> hiddenStatePair : distribHidden_States){
    distribHidden_States[hiddenStatePair.first] = (distribHidden_States[hiddenStatePair.first]/probReach)*probHist_;
  }*/
  //std::cout << "\n state evolve in : " << map_to_string(distribHidden_States);

  return distribHidden_States;
}


double MABCState::resolveRewards(const std::vector<Action>& actions){
  double sum = 0;
  /*
  std::cout << "\n for actions : " << actions << " in state  : "
   << distribHidden_States[hiddenState::s00] << " , " 
   <<distribHidden_States[hiddenState::s10] << " , "
   << distribHidden_States[hiddenState::s01] << " , " << distribHidden_States[hiddenState::s11] << std::flush;
   /**/
  /*
  SPIEL_CHECK_GE(distribHidden_States[hiddenState::s00]+distribHidden_States[hiddenState::s01]
    +distribHidden_States[hiddenState::s10]+distribHidden_States[hiddenState::s11], 0.999);
  SPIEL_CHECK_LT(distribHidden_States[hiddenState::s00]+distribHidden_States[hiddenState::s01]
    +distribHidden_States[hiddenState::s10]+distribHidden_States[hiddenState::s11], 1.001);*/
  //for (std::pair<hiddenState,double> hiddenStatePair : distribHidden_States){
    //hiddenState hiddenState_ = hiddenStatePair.first;

    //send, send : pass

    //send, wait  
      if (ToAction(actions[0]) == ActionType::kSend &&
          ToAction(actions[1]) == ActionType::kWait ){
            if (cur_state_ == hiddenState::s10){
              sum += 0.4;//*distribHidden_States.at(cur_state_);
            }
            if (cur_state_ == hiddenState::s11){
              sum += 0.4;//*distribHidden_States.at(cur_state_);
            }
      }
      if (ToAction(actions[0]) == ActionType::kWait &&
          ToAction(actions[1]) == ActionType::kSend ){
            if (cur_state_ == hiddenState::s01){
              sum += 0.4;//*distribHidden_States.at(cur_state_);
            }
            if (cur_state_ == hiddenState::s11){
              sum += 0.4;//*distribHidden_States.at(cur_state_);
            }
      }
    //}
    if (cur_state_==hiddenState::s00){
    std::cout << "\nsum of rewards : " << sum << " at timestep_ : " <<  timestep_ << " for actions : " << actions
    << " with state : " << "00";
    }
    if (cur_state_==hiddenState::s01){
    std::cout << "\nsum of rewards : " << sum << " at timestep_ : " <<  timestep_ << " for actions : " << actions
    << " with state : " << "01";
    }
    if (cur_state_==hiddenState::s10){
    std::cout << "\nsum of rewards : " << sum << " at timestep_ : " <<  timestep_ << " for actions : " << actions
    << " with state : " << "10";
    }
    if (cur_state_==hiddenState::s11){
    std::cout << "\nsum of rewards : " << sum << " at timestep_ : " <<  timestep_ << " for actions : " << actions
    << " with state : " << "11";
    }
    return sum;
}

std::vector<Action> MABCState::LegalActions(Player player) const {
  //std::cout << "\n player : " << player;
  //std::cout << "\n cur_player_ : " << cur_player_;
  if (player == 0 || player == 1) {
    return {0,1};
  } else if (IsTerminal()) {
    return {};
  } else //chance node
  {
    if (rollout_chance_==0){
      std::cout << "returning legal actions {-4,-3,-2,-1}: ";
      return {100,101,110,111};
    }
    else{
      std::cout << "returning legal actions 00,01,10,11: ";
      return {00,01,10,11};
    }
  }
}
std::string MABCState::map_to_string(std::map<hiddenState,double>  &m) const {
  std::string output = "";
  std::string convrt = "";
  std::string result = "";

  for (auto it = m.cbegin(); it != m.cend(); it++) {
    convrt = std::to_string(it->second);
    output += MABCState::stateToString(it->first) + ":" + (convrt) + ", ";
  }
  result = output.substr(0, output.size() - 2 );
  return result;
}


ActionsAndProbs MABCState::ChanceOutcomes() const {
  const std::vector<Action> & actions = lastActions_;

  if (rollout_chance_==0){
    std::cout << "\nhere";
    std::map<hiddenState,double> distribStates = evolveState(actions,lastActions_[0]);
    ActionsAndProbs a = {{100,distribStates.at(hiddenState::s00)},{101,distribStates.at(hiddenState::s01)},{110,distribStates.at(hiddenState::s10)},{111,distribStates.at(hiddenState::s11)}};
    std::cout<<"\n a : " << a;
    return a;
  }
  if (rollout_chance_==1){
    std::cout<<"\nthere";
  SPIEL_CHECK_TRUE(IsChanceNode());

  std::map<Action,double> distribObservation;
  //need to initialize it;

  
  //for (std::pair<hiddenState,double> hiddenStatePair : distribHidden_States){
    hiddenState hiddenState_ = cur_state_;// = hiddenStatePair.first;
    if (ToAction(actions[0]) == ActionType::kSend &&
          ToAction(actions[1]) == ActionType::kSend ){
        if (hiddenState_ == hiddenState::s00){
          distribObservation[0] += 0.81*distribHidden_States.at(hiddenState_);
          distribObservation[1] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[2] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[3] += 0.01*distribHidden_States.at(hiddenState_);
        }
        if (hiddenState_ == hiddenState::s01){
          distribObservation[0] += 0.81*distribHidden_States.at(hiddenState_);
          distribObservation[1] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[2] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[3] += 0.01*distribHidden_States.at(hiddenState_);
        }
        if (hiddenState_ == hiddenState::s10){
          distribObservation[0] += 0.81*distribHidden_States.at(hiddenState_);
          distribObservation[1] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[2] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[3] += 0.01*distribHidden_States.at(hiddenState_);
        }
        if (hiddenState_ == hiddenState::s11){
          distribObservation[0] += 0.81*distribHidden_States.at(hiddenState_);
          distribObservation[1] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[2] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[3] += 0.01*distribHidden_States.at(hiddenState_);
        }
      }
      else{
        //std::cout<< "\nplayers play Tail,Tail";
          if (hiddenState_ == hiddenState::s00){
            distribObservation[0] += 0.01*distribHidden_States.at(hiddenState_);
            distribObservation[1] += 0.09*distribHidden_States.at(hiddenState_);
            distribObservation[2] += 0.09*distribHidden_States.at(hiddenState_);
            distribObservation[3] += 0.81*distribHidden_States.at(hiddenState_);
          }
        if (hiddenState_ == hiddenState::s01){
          distribObservation[0] += 0.01*distribHidden_States.at(hiddenState_);
          distribObservation[1] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[2] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[3] += 0.81*distribHidden_States.at(hiddenState_);
        }
        if (hiddenState_ == hiddenState::s10){
          distribObservation[0] += 0.01*distribHidden_States.at(hiddenState_);
          distribObservation[1] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[2] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[3] += 0.81*distribHidden_States.at(hiddenState_);
        }
        if (hiddenState_ == hiddenState::s11){
          distribObservation[0] += 0.01*distribHidden_States.at(hiddenState_);
          distribObservation[1] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[2] += 0.09*distribHidden_States.at(hiddenState_);
          distribObservation[3] += 0.81*distribHidden_States.at(hiddenState_);
        }
      }
    return {{00,distribObservation[0]},{01,distribObservation[1]},{10,distribObservation[2]},{11,distribObservation[3]}};
  }
}

std::string MABCState::ToString() const {
  std::string result = "";
  absl::StrAppend(&result, "Total moves: ", total_moves_, "\n");
  absl::StrAppend(&result, "Most recent reward: ", reward_, "\n");
  absl::StrAppend(&result, "Total rewards: ", total_rewards_, "\n");

  return result;
}

bool MABCState::IsTerminal() const { 
  //return winner_ != kInvalidPlayer; 
  //std::cout << "is terminal ? timestep_ : " << std::to_string(timestep_);
  if (horizon_==timestep_){
    return true;
  }
  return false;
}

std::vector<double> MABCState::Returns() const {
  // Cooperative game: all players get same reward.
  return {reward_, -reward_};
}

std::vector<double> MABCState::Rewards() const {
  // Cooperative game: all players get same reward.
  return {reward_, -reward_};
}

std::unique_ptr<State> MABCState::Clone() const {
  return std::unique_ptr<State>(new MABCState(*this));
}

MABCGame::MABCGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      horizon_(ParameterValue<int>("horizon")),
      fully_observable_(ParameterValue<bool>("fully_observable")) {}

std::vector<int> MABCGame::ObservationTensorShape() const {
  if (fully_observable_) {
    return {3};
  } else {
    return {kNumObservations};
  }
}

int MABCGame::NumDistinctActions() const {
  return kNumDistinctActions;
}

int MABCGame::NumPlayers() const { return kNumPlayers; }

std::unique_ptr<State> MABCGame::NewInitialState() const {
  std::unique_ptr<State> state(
      new MABCState(shared_from_this(), horizon_, fully_observable_));
  return state;
}

double MABCGame::MaxUtility() const {
  return (MaxGameLength() *  kRMax);
}

double MABCGame::MinUtility() const {
  return (MaxGameLength() * kRMin);
}

}  // namespace MP
}  // namespace open_spiel
