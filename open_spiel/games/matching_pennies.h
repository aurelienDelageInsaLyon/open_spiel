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

#ifndef OPEN_SPIEL_GAMES_Matching_Pennies_H_
#define OPEN_SPIEL_GAMES_Matching_Pennies_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// This is the cooperative box-pushing domain presented by Seuken & Zilberstein
// in their paper "Improved Memory-Bounded Dynamic Programming for Dec-POMDPs"
// http://rbr.cs.umass.edu/papers/SZuai07.pdf
//
// Parameters:
//     "fully_observable" bool   agents see everything, or only partial view as
//                               described in the original paper (def: false)
//     "horizon"          int    length of horizon (default = 100)

namespace open_spiel {
namespace matching_pennies {


// When not fully-observable, the number of observations (taken from Seuken &
// Zilberstein '12): empty field, wall, other agent, small box, large box.
enum ObservationType {
  kNone,
};

// Different actions used by the agent.
enum class ActionType { kHead, kTail }; 

enum class hiddenState {si, sh, st};


class MatchingPenniesState : public SimMoveState {
 public:
  MatchingPenniesState(std::shared_ptr<const Game> game, int horizon,
                      bool fully_observable);
  std::string stateToString(hiddenState hiddenState_) const;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::map<hiddenState,double> distribHidden_States = {{hiddenState::si,1.0}, {hiddenState::st,0.0}, {hiddenState::sh,0.0}};

  //void ObservationTensor(Player player,
  //                       absl::Span<float> values) const override;
  //std::string ObservationString(Player player) const override;
  std::string InformationStateString(Player player) const override;

  std::vector<MatchingPenniesState::PlayerAction>& History(Player player) const;

  std::string map_to_string(std::map<hiddenState,double>  &m) const;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : cur_player_;
  }
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;

  void Reset(const GameParameters& params);
  std::vector<Action> LegalActions(Player player) const override;
  //void InformationStateTensor(Player player,
  //                            absl::Span<float> values) const override;
  double resolveRewards(const std::vector<Action>& actions);
  void evolveState(const std::vector<Action>& actions);
 protected:
  void DoApplyAction(Action action) override;
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  std::vector<Action> lastActions_;
  std::array<ActionType, 2> action_status_;
  hiddenState cur_state_;
  // Fields sets to bad/invalid values. Use Game::NewInitialState().
  double total_rewards_ = -1;
  Player cur_player_ = kSimultaneousPlayerId;
  int total_moves_ = 0;
  int initiative_;  // player id of player to resolve actions first.
  bool win_;        // True if agents push the big box to the goal.
  bool fully_observable_;
  int chanceMoves = 0;
  // Most recent rewards.
  double reward_;
  int num_players_;//this is handled by default construction with 2 players
  int horizon_;//added
  int timestep_ = 0;//added
  std::pair<std::vector<Action>,std::vector<Action>> possibleActions;//added
  std::pair<std::vector<Action>,std::vector<Action>> possibleObservations;//added
};

class MatchingPenniesGame : public SimMoveGame {
 public:
  explicit MatchingPenniesGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 2; }
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return horizon_; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }
  //std::vector<int> InformationStateTensorShape() const override;//todo

 private:
  int horizon_;
  bool fully_observable_;
};

}  // namespace coop_box_pushing
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COOP_BOX_PUSHING
