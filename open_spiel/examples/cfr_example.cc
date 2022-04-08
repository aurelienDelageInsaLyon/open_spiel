// Copyright 2021 DeepMind Technologies Limited
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

#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include <chrono>

// Example code for using CFR+ to solve Kuhn Poker.
int main(int argc, char** argv) {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGameAsTurnBased("posgFromFile");

  open_spiel::algorithms::CFRSolver solver(*game);
  std::cerr << "Starting CFR on " << game->GetType().short_name
            << "..." << std::endl;
  
  int i =0;
  auto start = std::chrono::steady_clock::now();
  auto time_left = std::chrono::steady_clock::now();
  float val = 0.0;
  
  while (i<10){
    solver.EvaluateAndUpdatePolicy();

    i++;
    auto time_left = std::chrono::steady_clock::now();
    val = std::chrono::duration_cast<std::chrono::nanoseconds>(time_left - start).count()*1e-9;
    //std::cout << "time taken : " << val;
  }

    double exploitability = open_spiel::algorithms::Exploitability(
          *game, *solver.AveragePolicy());

    std::cout << "Iteration " << i << " exploitability=" << exploitability;
    std::cout << "\n\n average policy : " << solver.TabularAveragePolicy().ToString();
}
