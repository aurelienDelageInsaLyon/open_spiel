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

#include "open_spiel/games/posg_from_file.h"

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
namespace posgFromFile {
namespace {

int kNumPlayers = 2;
int kNumDistinctActions = 2;
int kNumStates = 2;
int kNumObservations = 4;


// Rewards.
constexpr double kRMax = 3;
constexpr double kRMin = -6;

// Default parameters.
constexpr int kDefaultHorizon = 3;
constexpr bool kDefaultFullyObservable = false;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"posgFromFile",
    /*long_name=*/"posgFromFile",
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
  return std::shared_ptr<const Game>(new posgFromFileGame(params));
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
  for (int i = 0; i< nbActions[0]; i++){
    if (action==i)
      return std::to_string(i);
  }
  for (int i = 0; i< nbActions[1]; i++){
    if (action==i)
      return std::to_string(i);
  }
  SpielFatalError(absl::StrCat("Invalid action: ", action));
}


std::vector<Action> observationToAction(Action z) {
  //std::cout << " z : " << z << " player : " << player;
  switch(z){
    case 10:
      return {0,0};
    case 11:
      return {0,1};
    case 12:
      return {1,0};
    case 13:
      return {1,1};
  }
  return {98,99};
}

std::string observationToString(Action z) {
  SpielFatalError(absl::StrCat("Invalid observation: ", z));
  return "bla";
  for (int i = 0; i< nbObservations[0]; i++){
    if (z==i)
      return std::to_string(i);
  }
  for (int i = 0; i< nbObservations[1]; i++){
    if (z==i)
      return std::to_string(i);
  }

}

std::map<int,double> deep_copy(std::map<int,double> tmpMap)
{
    std::map<int, double>::iterator it;
    std::map<int,double> res;
    for (int i = 0; i< tmpMap.size(); i++)
    {
      res[i] = tmpMap[i];
    }
    return res;
}

}  // namespace

  posgFromFileState::posgFromFileState(std::shared_ptr<const Game> game,
                                         int horizon, bool fully_observable,
                                         std::map<std::pair<int,std::vector<Action>>,std::map<int,double>> transitions,
                                         std::map<std::pair<std::vector<Action>,std::vector<Action>>,std::map<int,double>> observations,
                                         std::map<std::pair<int,std::vector<Action>>,double> rewards,
                                         std::map<int,double> initialDistrib)
    : SimMoveState(game),
      horizon_(horizon),
      cur_player_(kSimultaneousPlayerId),
      total_moves_(0),
      initiative_(0),
      win_(false),
      fully_observable_(fully_observable),
      reward_(0),
      transitionFunction(transitions),
      observationFunction(observations),
      rewardFunction(rewards),
      distribHidden_States(initialDistrib),
      lastDistribHidden_States(initialDistrib)
    {
    }


std::string posgFromFileState::ActionToString(Player player, Action action) const {
  return std::to_string(action);
}
std::string posgFromFileState::observationToString(Action z,Player player) const {
  std::cout << " z : " << z << " player : " << player;
  if (player==0){
      int a = z-10;
      a = (z-10)/2;//10 -> 0, 11 -> 0, 12-> 1, 13 -> 1
      //std::cout << " \n  obs transf : " << z << "a : " << a;
      return ("z" + std::to_string(a) + "p0");
  }
  else{
      int a = z-10;
      a = z%nbObservations[0];//10->0, 11 -> 1, 12->0, 13->1
      //std::cout << " \n  obs transf : " << z << "a : " << a;
      return ("z" + std::to_string(a) + "p1");
  }
  SpielFatalError(absl::StrCat("Invalid observation: ", z));

}

std::string posgFromFileState::actionToString(Action action,Player player) const {
  switch (action) {
    case 0:
      return ("a0p"+std::to_string(player));
    case 1:
      return ("a1p"+std::to_string(player));
    case 2:
      return ("a2p"+std::to_string(player));
  }
  SpielFatalError(absl::StrCat("Invalid action: ", action));
  return "invalid action";
}
std::string posgFromFileState::stateToString(hiddenState hiddenState_) const {
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

std::string posgFromFileState::InformationStateString(Player player) const {
  const posgFromFileGame& game = open_spiel::down_cast<const posgFromFileGame&>(*game_);
  //return posgFromFileState::History();
  //std::cout << "\ncall information state string with player : " << player;
  std::vector<posgFromFileState::PlayerAction> hist = posgFromFileState::FullPrivateHistory(player);
  std::vector<std::string> histString;
  std::transform(std::begin(hist),
               std::end(hist), 
               std::back_inserter(histString),
               [this,&player](posgFromFileState::PlayerAction d) { if (d.player==-1){ return observationToString(d.action,player);} else {return actionToString(d.action,d.player);} } 
              );
  //std::cout << "\n \n i'm returning :  " << std::accumulate(histString.begin(), histString.end(), std::string(""));
  return std::accumulate(histString.begin(), histString.end(), std::string(""));
}

void posgFromFileState::DoApplyActions(const std::vector<Action>& actions) {
  SPIEL_CHECK_EQ(actions.size(), 2);
  SPIEL_CHECK_EQ(cur_player_, kSimultaneousPlayerId);
  //std::cout << "simultaneous play \n";
  if (IsSimultaneousNode()){ 
    evolveState(lastActions_);
    reward_+= posgFromFileState::resolveRewards(actions);
    cur_player_ = kChancePlayerId;
    //std::cout << "\n players played : " << actions << " at timestep " << timestep_<<"\n";
    timestep_++;
    lastActions_ = actions;
    for (int i =0;i<nbStates;i++){
      lastDistribHidden_States[i] = distribHidden_States[i];
    }
    //lastDistribHidden_States = {{hiddenState::s00,distribHidden_States[hiddenState::s00]},{hiddenState::s01,distribHidden_States[hiddenState::s01]},
    //      {hiddenState::s10,distribHidden_States[hiddenState::s10]},{hiddenState::s11,distribHidden_States[hiddenState::s11]}};
    total_moves_++;
  }
}

void posgFromFileState::DoApplyAction(Action action) {
  if (IsSimultaneousNode()){
    std::cout << "weird, shouldn't be a simultaneous node";
    return;
  }
  
  //chance node
  else{
    //std::cout << "\n chance plays " << action << " at timestep : " << timestep_;
    cur_player_ = kSimultaneousPlayerId;
    lastObservations=action;
  }
}

void posgFromFileState::evolveState(const std::vector<Action>& actions){
  std::cout << "\nevolving state\n" << std::flush;
  if (timestep_==0){
    std::cout << "\n returning\n" << std::flush;  
    return;
  }
  Action lastJointObservation = lastObservations;
  std::vector<std::pair<Action, double>> valoutcomes = ChanceOutcomes();
  double val = 0.0;
  for (int i = 0;i<valoutcomes.size();i++){
    if (valoutcomes[i].first==lastObservations){
      val=valoutcomes[i].second;
    }
  }
  probHist_*=val;
  std::cout << "\n val last obs : " << val << " actions : " << actions << " last obs : " << lastObservations;
  //std::cout << "\n initialDistrib : " << distribHidden_States[0] << ", " << distribHidden_States[1] << ", " << distribHidden_States[2] << ", " << distribHidden_States[3];
  //std::cout << "distribHidden_States : " << distribHidden_States[0] << distribHidden_States[1] << distribHidden_States[2];
  std::map<int,double> res;
  for (int nextS = 0; nextS<nbStates;nextS++){
    double proba = 0;
    for (int s = 0; s<nbStates;s++){
      //std::cout << "\n value transition : " << transitionFunction.at(std::make_pair(s,std::vector<Action>(actions))).at(nextS);
      //std::cout << "\n lastObservations : " <<observationToAction(lastObservations) << " from : " << lastObservations << std::flush;
        proba+=lastDistribHidden_States.at(s)
              *transitionFunction.at(std::make_pair(s,std::vector<Action>(actions))).at(nextS)
              *observationFunction.at(std::make_pair(observationToAction(lastObservations),actions)).at(nextS);
    }
    //std::cout << "\nproba state : " << proba;
    if (val>0.0){
      proba = proba/val;
      res.insert(std::make_pair(nextS,proba));
    }
    else{
      res.insert(std::make_pair(nextS,0.0));
    }
  }
  distribHidden_States = res; 
  std::cout << "\n actions : " << actions <<"evolves in : "  << distribHidden_States[0] << ", " << distribHidden_States[1] 
  << ", " << distribHidden_States[2] <<", "<< distribHidden_States[3] << "with last obs : " << lastObservations << "\n" << std::flush;
}


double posgFromFileState::resolveRewards(const std::vector<Action>& actions){
  if (timestep_==1 || timestep_==0){
  std::cout << "\n distrib in get rewards: " << distribHidden_States[0] << distribHidden_States[1]<<"\n";
  }
  double res = 0;
  for (int s=0;s<nbStates;s++){
    //if (rewardFunction.count(std::make_pair(s,actions))==1){
    if (distribHidden_States[s]>0.0 && distribHidden_States[s] < kRMax*100){
      std::cout << "\ndistribHidden_States[s]" <<distribHidden_States[s];
      res +=distribHidden_States[s]*rewardFunction.at(std::make_pair(s,actions))*nbObservations[0]*nbObservations[1];
    }
    //}
    /*
    else{
      std::cout << "\n shouldn't happen!";
      std::exit(1);
      return 0.0;
    }*/
  }
  std::cout << "\ncomputed rewards : " << res << " for actions : " << actions << "in state : " << distribHidden_States[0] << "," << distribHidden_States[1];
  return res;
}

std::vector<Action> posgFromFileState::LegalActions(Player player) const {
  //std::cout << "\n player : " << player;
  //std::cout << "\n cur_player_ : " << cur_player_;
  if (player == 0){
    std::vector<Action> res = std::vector<Action>(nbActions[0]);
    std::iota(res.begin(),res.end(),0);
    return res;
  }
  if (player == 1){
    std::vector<Action> res = std::vector<Action>(nbActions[1]);
    std::iota(res.begin(),res.end(),0);
    return res;
  }
  else if (IsTerminal()) {
    return {};
  } else{ //chance node
      std::vector<Action> res = std::vector<Action>(nbObservations[0]*nbObservations[1]);
      std::iota(res.begin(),res.end(),10);
      //std::cout << "res : " << res;
      return res;
    }
}
std::string posgFromFileState::map_to_string(std::map<hiddenState,double>  &m) const {
  std::string output = "";
  std::string convrt = "";
  std::string result = "";

  for (auto it = m.cbegin(); it != m.cend(); it++) {
    convrt = std::to_string(it->second);
    output += posgFromFileState::stateToString(it->first) + ":" + (convrt) + ", ";
  }
  result = output.substr(0, output.size() - 2 );
  return result;
}


ActionsAndProbs posgFromFileState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> res;
  std::cout << "\n distrib : " << distribHidden_States.at(0) << "," << distribHidden_States.at(1);
  for (int z0 = 0; z0<nbObservations[0];z0++){
    for (int z1 = 0; z1<nbObservations[1];z1++){
      double proba = 0;
      for (int nextS = 0; nextS<nbStates;nextS++){
        for (int s = 0; s<nbStates;s++){
          proba+=distribHidden_States.at(s)*transitionFunction.at(std::make_pair(s,lastActions_)).at(nextS)
                *observationFunction.at(std::make_pair(std::vector<Action>({z0,z1}),std::vector<Action>(lastActions_))).at(nextS);
        }
      }
      //std::cout << "\nproba observations : " << proba;
      res.push_back(std::make_pair(10+z0*nbObservations[1]+z1,proba));
    }
  }
  std::cout << "\nchance observations :" << res[0].second <<","<<res[1].second <<","<< res[2].second <<","<< res[3].second
    << " in state : " << distribHidden_States.at(0) << "," << distribHidden_States.at(1)<< " actions done : " << lastActions_ <<"\n"<<std::flush;
  return res;
}

std::string posgFromFileState::ToString() const {
  std::string result = "";
  absl::StrAppend(&result, "Total moves: ", total_moves_, "\n");
  absl::StrAppend(&result, "Most recent reward: ", reward_, "\n");
  absl::StrAppend(&result, "Total rewards: ", total_rewards_, "\n");

  return result;
}

bool posgFromFileState::IsTerminal() const { 
  //return winner_ != kInvalidPlayer; 
  //std::cout << "is terminal ? timestep_ : " << std::to_string(timestep_);
  if (horizon_==timestep_){
    return true;
  }
  return false;
}

std::vector<double> posgFromFileState::Returns() const {
  // Cooperative game: all players get same reward.
  return {reward_, -reward_};
}

std::vector<double> posgFromFileState::Rewards() const {
  // Cooperative game: all players get same reward.
  return {reward_, -reward_};
}

std::unique_ptr<State> posgFromFileState::Clone() const {
  //std::cout << "\ntrying to clone..." << std::flush;
  std::unique_ptr<State> a = std::unique_ptr<State>(new posgFromFileState(*this));
  //a->transitionFunction = transitionFunction;
  //a->observationFunction = observationFunction;
  //a->rewardFunction = rewardFunction;
  //std::cout << " transition function.size() : " << a->transitionFunction.size();
  return a;
}

posgFromFileGame::posgFromFileGame(const GameParameters& params)
    : SimMoveGame(kGameType, params),
      horizon_(ParameterValue<int>("horizon")),
      fully_observable_(ParameterValue<bool>("fully_observable")) {
    
    std::cout << "----------------------------------"<<std::endl;
    std::cout << "-         TestParsing()          -"<<std::endl;
    std::cout << "----------------------------------"<<std::endl;
    
    DecPOMDPDiscrete test("Test parse problem", "parses the test.dpomdp file ", "recycling.dpomdp");
    
    bool printing = true;
    try{
        //std::cout << " testing \"test.MultiAgentDecisionProcess::Print()\"... :\n";
        test.MultiAgentDecisionProcess::Print();
        std::cout << std::endl << "<" <<std::endl;
        //std::cout << " testing \"test.GetProblemFile()\"... :\n";
        std::cout << test.GetProblemFile();
        std::cout << std::endl << "<" <<std::endl;
        
        test.Print();
    }
    catch (E& e){e.Print();}

    try{
        //std::cout << "\n>>PARSE TEST: test" << std::endl;
        MADPParser dpomdpd_parser_test (&test);
    }
    catch (E& e){e.Print();}
    try{test.Print();}
    catch (E& e){e.Print();}

    nbStates = static_cast<int>(test.GetNrStates());
    nbActions = {static_cast<int>(test.GetNrActions(Index(0))),static_cast<int>(test.GetNrActions(Index(1)))};
    nbObservations = {static_cast<int>(test.GetNrObservations(Index(0))),static_cast<int>(test.GetNrObservations(Index(1)))};

    kNumPlayers=2;
    kNumStates=nbStates;
    kNumDistinctActions=nbActions[0]*nbActions[1];

    /*
    std::cout << "\n \n nb states : " << nbStates << " access " << test.GetReward(Index(1),Index(0)) << std::flush;
    std::cout << "\n \n nb actions P1 : " << nbActions[0] << " P2 : " << nbActions[1];
    std::cout << "\n \n nb observations P1 : " << nbObservations[0] << " P2 : " << nbObservations[1];
    */
    //std::cout << "\n transition : " << test.GetTransitionProbability(Index(0),Index(0),Index(0)) << std::flush;
    //std::cout << "\n observation : " << test.GetObservationProbability(Index(0),Index(0),Index(0)) << std::flush;
    for (int i = 0; i<nbStates;i++){
      distribHidden_States.insert({i,test.GetInitialStateProbability(Index(i))});
      lastDistribHidden_States.insert({i,test.GetInitialStateProbability(Index(i))});
    }
  if (printing){
    std::cout << "\n ----------------- observation function : --------------------\n";
  }
  std::map<int,double> tmpMapObservation;
  for (int z0 = 0; z0<nbObservations[0]; z0++){
    for (int z1 = 0; z1<nbObservations[1]; z1++){
      for (int i = 0; i<nbActions[0]; i++){
        for (int j = 0; j<nbActions[1]; j++){
          tmpMapObservation.clear();
          for (int nextS = 0; nextS<nbStates; nextS++){
            tmpMapObservation.insert({nextS,test.GetObservationProbability(Index(i*nbActions[1]+j),Index(nextS),Index(z0*nbObservations[1]+z1))});
          }
          std::map<int,double> tmpMapCopyObservation = deep_copy(tmpMapObservation);
          resObservation.insert(std::make_pair(std::make_pair(std::vector<Action>({z0,z1}),std::vector<Action>({i,j})),tmpMapCopyObservation));
        }
      }
    }
  }
  //observationFunction = resObservation;
for (int z0 = 0; z0<nbObservations[0]; z0++){
    for (int z1 = 0; z1<nbObservations[1]; z1++){
      for (int i = 0; i<nbActions[0]; i++){
        for (int j = 0; j<nbActions[1]; j++){
          if (printing){
        std::cout << "\n z0 : " << z0 << " z1 : " << z1 << "i,j : " << i << ", " << j << " : " << resObservation[std::make_pair(std::vector<Action>({z0,z1}),std::vector<Action>({i,j}))][0] 
          << resObservation[std::make_pair(std::vector<Action>({z0,z1}),std::vector<Action>({i,j}))][1] << resObservation[std::make_pair(std::vector<Action>({z0,z1}),std::vector<Action>({i,j}))][2] 
          << resObservation[std::make_pair(std::vector<Action>({z0,z1}),std::vector<Action>({i,j}))][3]<<std::flush;
          }
      }
    }
    }
  }
  if (printing){
  std::cout << " \n ---------------- end observation function ------------------------- \n"<<std::flush;

  std::cout << " \n ---------------- reward function ------------------------- \n" <<std::flush;
  }
  double val;
  for (int s = 0; s<nbStates; s++){
      for (int i = 0; i<nbActions[0]; i++){
        for (int j = 0; j<nbActions[1]; j++){
          val = test.GetReward(Index(s),Index(i*nbActions[1]+j));
          resReward.insert(std::make_pair(std::make_pair(s,std::vector<Action>({i,j})),val));
        }
      }
  }
  for (int s = 0; s<nbStates; s++){
    for (int i = 0; i<nbActions[0]; i++){
      for (int j = 0; j<nbActions[1]; j++){
        if (printing){
        std::cout << "\n s  : " << s << "i,j" << i << ", " <<j << " : " << resReward[std::make_pair(s,std::vector<Action>({i,j}))] << std::flush;
        }
      }
    }
  }
  //rewardFunction = resReward;
          if (printing){

  std::cout << " \n ---------------- end reward function ------------------------- \n";  
  }
    //problem = test;

    /*
    setRewardFunction();
    setTransitionFunction();
    setObservationFunction();
    */
        if (printing){

  std::cout << " in get transition function \n\n\n";
}
  std::map<int,double> tmpMap;
  for (int s = 0; s<nbStates; s++){
    for (int i = 0; i<nbActions[0]; i++){
      for (int j = 0; j<nbActions[1]; j++){
        tmpMap.clear();
        for (int nextS = 0; nextS<nbStates; nextS++){
          tmpMap.insert({nextS,test.GetTransitionProbability(Index(s),Index(i*nbActions[1]+j),Index(nextS))});
        }
        std::map<int,double> tmpMapCopy = deep_copy(tmpMap);
        res.insert(std::make_pair(std::make_pair(s,std::vector<Action>({i,j})),tmpMapCopy));
      }
    }
  }
  //transitionFunction = res;
          if (printing){

  std::cout << "\n ----------------- transition function : --------------------\n";
}
  for (int s = 0; s<nbStates; s++){
    for (int i = 0; i<nbActions[0]; i++){
      for (int j = 0; j<nbActions[1]; j++){
                if (printing){

        std::cout << "\n s : " << s << "i,j : " << i << ", " << j << " : " << res[std::make_pair(s,std::vector<Action>({i,j}))][0] 
          << res[std::make_pair(s,std::vector<Action>({i,j}))][1] << res[std::make_pair(s,std::vector<Action>({i,j}))][2]<<std::flush;
        }
      }
    }
  }
          if (printing){

  std::cout << " \n ---------------- end transition function ------------------------- \n";
}        if (printing){

    for (auto& t : distribHidden_States)
    std::cout << "\n distribHiddenStates : " << t.first << " " 
              << t.second << "\n" ; 
    //distribHidden_States = {{hiddenState::s00,0.0}, {hiddenState::s01,0.0}, {hiddenState::s10,0.0},{hiddenState::s11,1.0}};
            }
      }

std::vector<int> posgFromFileGame::ObservationTensorShape() const {
  if (fully_observable_) {
    return {3};
  } else {
    return {kNumObservations};
  }
}

int posgFromFileGame::NumDistinctActions() const {
  return kNumDistinctActions;
}

int posgFromFileGame::NumPlayers() const { return kNumPlayers; }

std::unique_ptr<State> posgFromFileGame::NewInitialState() const {

  std::unique_ptr<State> state(
      new posgFromFileState(shared_from_this(), horizon_, fully_observable_,res,resObservation,resReward,distribHidden_States));
  return state;
}

double posgFromFileGame::MaxUtility() const {
  return (MaxGameLength() *  kRMax);
}

double posgFromFileGame::MinUtility() const {
  return (MaxGameLength() * kRMin);
}

}  // namespace MP
}  // namespace open_spiel
