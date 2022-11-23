/**
 * @file HsviMG.hpp
 * @author David Albert (david.albert@insa-lyon.fr)
 * @brief HsviMG algorithm
 * @version 0.1
 * @date 22/12/2020
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#pragma once

#include <string>

#include <sdm/types.hpp>
#include <sdm/core/state/state.hpp>
#include <sdm/core/action/action.hpp>
#include <sdm/algorithms/planning/tsvi.hpp>
#include <sdm/world/solvable_by_hsvi.hpp>
#include <sdm/utils/value_function/value_function.hpp>
#include <sdm/core/action/stochastic_decision_rule.hpp>
#include <sdm/core/state/occupancy_state_mg.hpp>
#include <sdm/utils/linear_algebra/hyperplane/vectorMG.hpp>
#include <sdm/core/state/history_tree.hpp>

namespace sdm
{

	/**
   	 * @brief [Heuristic Search Value Iteration (HsviMG)](https://arxiv.org/abs/1207.4166) 
   	 * and its extensions (FB-HsviMG, one-sided HsviMG).
   	 * 
   	 * This class contains the general algorithmic scheme of HsviMG. By redefining 
   	 * the lower and upper bounds or the way the world is represented, we will be 
   	 * able to obtain different state-of-the-art algorithms such as: HsviMG, oHsviMG, 
   	 * feature-based HsviMG, one-sided HsviMG, etc.
   	 * 
   	 */
	class HsviMG : public TSVI,
				 public std::enable_shared_from_this<HsviMG>
	{
	public:
		/**
    	 * @brief Construct the HsviMG algorithm.
    	 * 
    	 * @param world the problem to be solved by HsviMG
    	 * @param lower_bound the lower bound 
    	 * @param upper_bound the upperbound
    	 * @param planning_horizon the planning horizon
    	 * @param epsilon the error
    	 * @param num_max_trials the maximum number of trials before stop
    	 * @param name the name of the algorithm (this name is used to save logs)
    	 */
		HsviMG(std::shared_ptr<SolvableByHSVI> &world,
			 std::shared_ptr<ValueFunction> lower_bound,
			 std::shared_ptr<ValueFunction> upper_bound,
			 double error,
			 number num_max_trials = 1000000,
			 std::string name = "HsviMG",
			 number lb_update_frequency = 1,
			 number ub_update_frequency = 1,
			 double time_max = 1000000,
			 bool keep_same_action_forward_backward = false);

		int agent_id_ = 0;
		void initialize();

		/**
    	 * @brief Check the end of HsviMG algo.
    	 * 
    	 * @param state the current state
    	 * @param cost_so_far the cost so far
    	 * @param h the current timestep
    	 * @return true if optimal is reached or number of trials is bigger than maximal number of trials
    	 * @return false elsewhere
    	 */
		bool stop(const std::shared_ptr<State> &state, double cost_so_far, number h);

		/**
    	 * @brief Computes the error between bounds (or excess).
    	 * 
    	 * @param the state
    	 * @param cost the cost so far
    	 * @param h the timestep
    	 * @return the error
    	 */
		double excess(const std::shared_ptr<State> &state, double cost_so_far, number h);
		
		void explore(const std::shared_ptr<State> &state, double cost_so_far, number t);

		/**
		 * @brief Log execution variables in output streams.
		 */
		void logging();

		/**
         * @brief Get the name of the algorithm as a string. 
         * 
         * This function will return the name of the algorithm as a string. 
         * It does not return the name of a specific instance (`name` attribute) 
         * but those of the general algorithm used (i.e. HsviMG, QLearning, etc).
         * 
         * @return the algorithm name 
         */
		std::string getAlgorithmName();

		/**
    	 * @brief Get the lower bound value function 
    	 */
		std::shared_ptr<ValueFunction> getLowerBound() const;

		/**
    	 * @brief Get the upper bound value function 
    	 */
		std::shared_ptr<ValueFunction> getUpperBound() const;

		/**
         * @brief Update the value function at a specific state and time step.
         * 
         * This function will make an update on the value function using the 
         * bellman operator. this update is done for a particular state and time step.
         * 
         * @param state the state 
         * @param t the time step
         */
		void updateValue(const std::shared_ptr<State> &state, number t);
		
		void updateLobValue(const std::shared_ptr<State> &state, number t, const std::shared_ptr<State> &lastState, const std::shared_ptr<StochasticDecisionRule> & dr);
		void updateUpbValue(const std::shared_ptr<State> &state, number t, const std::shared_ptr<State> &lastState, const std::shared_ptr<StochasticDecisionRule> & dr);


		void saveParams(std::string filename, std::string format = ".md");

		void saveResults(std::string filename, std::string format = ".md");

		void updateLastStep(const std::shared_ptr<State> & state, const std::shared_ptr<State> & last_state,std::map<std::shared_ptr<HistoryInterface>,double> valuesUpb, std::map<std::shared_ptr<HistoryInterface>,double> valuesLob,const std::shared_ptr<StochasticDecisionRule> & lastDr1,const std::shared_ptr<StochasticDecisionRule> & lastDr2,const std::shared_ptr<StochasticDecisionRule> & dr1,const std::shared_ptr<StochasticDecisionRule> & dr2, number t);

		void explore(const std::shared_ptr<State> &last_state, const std::shared_ptr<StochasticDecisionRule> & dr1,const std::shared_ptr<StochasticDecisionRule> & dr2, const std::shared_ptr<State> &state, double cost_so_far, number t);
		
		std::shared_ptr<StochasticDecisionRule> extractStrategy(const std::shared_ptr<VectorMG> & vector, int horizon, int agent);

		std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> recGetRWCat(int t, const std::tuple<std::shared_ptr<OccupancyStateMG>,std::shared_ptr<StochasticDecisionRule>,std::shared_ptr<VectorMG>> & deltaStrat, int horizon);

		std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double> recGetRWMix(int t, const std::shared_ptr<VectorMG> & vector, int horizon);

		bool isExpansion(std::shared_ptr<HistoryTree> h1, std::shared_ptr<HistoryTree> h2);

		std::shared_ptr<HistoryTree> revertHistory(std::shared_ptr<HistoryTree> head, std::shared_ptr<HistoryTree> tail);

		std::shared_ptr<Action> getRevertLastAction(std::shared_ptr<HistoryTree> h);

		//std::shared_ptr<HistoryTree> addBeginningToHistory(const std::shared_ptr<HistoryTree> & h, const std::shared_ptr<Action>&  action, const std::shared_ptr<Observation> & observation);
		std::shared_ptr<Action> action_stored_rec;
		std::shared_ptr<Observation> obs_stored_rec;
		std::shared_ptr<HistoryTree> tmpH;

		int horizon;

		int testSolution(std::shared_ptr<VectorMG> vecUpb, std::shared_ptr<VectorMG> vecLob);

	protected:
		/**
         * @brief Initialize the loggers.
         * 
         * The loggers are objects allowing to write the statistics relative to the execution of the 
         * algorithm in files. The files in question can have different formats (text, XML, CSV, etc.) 
         * depending on the desired processing afterwards.
         * 
         */
		void initLogger();

		/**
         * @brief Initialize a trial.
         * 
         * This function will initialize arguments required in a trial.
         * 
         */
		void initTrial();

		/**
         * @brief Select the list of actions to explore.
         * 
         * This function can be inherited to build algorithms with different 
         * exploration heuristics
         * 
         * @return a list of actions 
         */
		std::vector<std::shared_ptr<Action>> selectActions(const std::shared_ptr<State> &state, number t);

		/**
         * @brief Select the list of observations to explore.
         * 
         * This function can be inherited to build algorithms with different 
         * exploration heuristics
         * 
         * @return a list of observations 
         */
		std::vector<std::shared_ptr<Observation>> selectObservations(const std::shared_ptr<State> &state, const std::shared_ptr<Action> &action, number t);

		/** @brief Lower Bound representation. */
		std::shared_ptr<ValueFunction> lower_bound, upper_bound;

		/** @brief Some variables for the algorithm. */
		number num_max_trials, lb_update_frequency, ub_update_frequency;

		double time_max, duration;

		bool keep_same_action_forward_backward;

		std::chrono::high_resolution_clock::time_point start_time, current_time;

		std::shared_ptr<OccupancyStateMG> lastState;
		StochasticDecisionRule lastStratP1;
		StochasticDecisionRule lastStratP2;

		std::map<int,std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double>> rwForTimestep = *(new std::map<int,std::map<std::tuple<std::shared_ptr<HistoryInterface>,std::shared_ptr<HistoryInterface>>,double>>());

	};
} // namespace sdm
