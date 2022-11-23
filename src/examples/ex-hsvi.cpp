
#include <cstdlib>
#include <iostream>

#include <sdm/config.hpp>
#include <sdm/exception.hpp>
#include <sdm/parser/parser.hpp>

#include <sdm/world/solvable_by_hsvi.hpp>
#include <sdm/world/occupancy_mdp.hpp>
#include <sdm/algorithms/planning/hsvi_mg.hpp>
#include <sdm/utils/value_function/action_selection/lp/action_maxplan_lp_mg.hpp>
#include <sdm/utils/value_function/update_operator/vupdate/tabular_update.hpp>
#include <sdm/utils/value_function/vfunction/tabular_value_function.hpp>
#include <sdm/utils/value_function/vfunction/partial_Q_value_function.hpp>
#include <sdm/utils/value_function/update_operator/vupdate/approxW_update.hpp>
#include <sdm/world/private_occupancy_mdp.hpp>


using namespace sdm;

int main(int argc, char **argv)
{
	std::string filename = "/home/delage/gitSdms3/sdms/data/world/dpomdp/matching_pennies.dpomdp";
	number horizon = 3, truncation = 0;
	double error = 0.0, discount = 1.;
	try
	{
		int agent_id_=0;
		// Parse file into MPOMDP
		auto mdp = sdm::parser::parse_file(filename);
		mdp->setHorizon(horizon);
		mdp->setDiscount(discount);

		// Instanciate the problem
		std::shared_ptr<SolvableByHSVI> hsvi_mdp = std::make_shared<OccupancyMG>(mdp, (truncation > 0) ? truncation : horizon, true, true);

		// Instanciate Initializer
		auto lb_init = std::make_shared<MinInitializer>(hsvi_mdp);
		auto ub_init = std::make_shared<MaxInitializer>(hsvi_mdp);

		// Instanciate action selection 
		std::shared_ptr<ActionSelectionInterface> action_selection_ub = std::make_shared<ActionSelectionMaxplanLPMG>(hsvi_mdp,0);

		std::shared_ptr<ActionSelectionInterface> action_selection_lb = std::make_shared<ActionSelectionMaxplanLPMG>(hsvi_mdp,1);
		// Declare bounds
		std::shared_ptr<ValueFunction> lb, ub;

		// Instanciate lower bound
		lb = std::make_shared<partialQValueFunction>(hsvi_mdp, lb_init, action_selection_lb,1);

		// Instanciate lower bound update operator
		auto lb_update_operator = std::make_shared<UpdateW>(lb);
		//auto lb_update_operator = std::make_shared<TabularUpdate>(lb);
		lb->setUpdateOperator(lb_update_operator);

		// Instanciate upper bound
		ub = std::make_shared<partialQValueFunction>(hsvi_mdp, ub_init, action_selection_ub, 0);

		// Instanciate upper bound update operator
		auto ub_update_operator = std::make_shared<UpdateW>(ub);
		//auto ub_update_operator = std::make_shared<TabularUpdate>(ub);
		ub->setUpdateOperator(ub_update_operator);

		lb->initialize();
		ub->initialize();
		
		// Instanciate HSVI
		auto algo = std::make_shared<HsviMG>(hsvi_mdp, lb, ub, error, 100000000, "", 1, 1,1000000);
        //std::cout << "\n value ub afer init : " << ub->getValueAt(hsvi_mdp->getInitialState(),0);
		//std::cout << "\n value lb afer init : " << lb->getValueAt(hsvi_mdp->getInitialState(),0);
		//std::exit(1);
		// Initialize and solve the problem
		algo->initialize();

		//lb->initialize();
		//std::cout << "\n value lb afer init : " << lb->getValueAt(hsvi_mdp->getInitialState(),0);
        //std::exit(1);
		std::cout<< Belief::PRECISION;
		std::cout<< VectorMG::PRECISION;
		//std::exit(1);
		algo->solve();


		error=0.0;
		///////////////////////////////////// computing BR P2 ///////////////////////////////////////////
		
		std::shared_ptr<VectorMG> vecUpb = std::dynamic_pointer_cast<partialQValueFunction>(ub)->getSupportVector(1);
		//std::cout << "\n vec : " << vecUpb->str();
		
		std::shared_ptr<StochasticDecisionRule> extractedStrategy = algo->extractStrategy(vecUpb,horizon,0);
		
		std::cout << "\n extracted strategy : " << extractedStrategy->str();
		//std::exit(1);
		int solveBR = 1;
		if (solveBR){
			//now solve the BR 
			// Parse file into MPOMDP

			const number num_player_ = 0;
			// Instanciate the problem
			std::shared_ptr<SolvableByHSVI> hsvi_mdp_br = std::make_shared<PrivateOccupancyMDP>(mdp, extractedStrategy, num_player_,-1, true, true);

			// Instanciate Initializer
			auto lb_init_br = std::make_shared<MinInitializer>(hsvi_mdp_br);
			auto ub_init_br = std::make_shared<MaxInitializer>(hsvi_mdp_br);

			// Instanciate action selection 
			auto action_tabular_br = std::make_shared<ExhaustiveActionSelection>(hsvi_mdp_br);

			// Declare bounds
			std::shared_ptr<ValueFunction> lb_br, ub_br;

			// Instanciate lower bound
			lb_br = std::make_shared<TabularValueFunction>(hsvi_mdp_br, lb_init_br, action_tabular_br);

			// Instanciate lower bound update operator
			auto lb_update_operator_br = std::make_shared<TabularUpdate>(lb_br);
			lb_br->setUpdateOperator(lb_update_operator_br);

			// Instanciate upper bound
			ub_br = std::make_shared<TabularValueFunction>(hsvi_mdp_br, ub_init_br, action_tabular_br);

			// Instanciate upper bound update operator
			auto ub_update_operator_br = std::make_shared<TabularUpdate>(ub_br);
			ub_br->setUpdateOperator(ub_update_operator_br);
			
			// Instanciate HSVI
			auto algo_br = std::make_shared<HSVI>(hsvi_mdp_br, lb_br, ub_br, error, 10000, "", 1, 1);
			algo_br->agent_id_ = (num_player_*2)-1;
			
			//auto algo = std::make_shared<ValueIteration>(hsvi_mdp,lb,error,100000,"");

			// Initialize and solve the problem
			algo_br->initialize();
			algo_br->solve();
			//std::exit(1);
		}
		//std::exit(1);
		
		///////////////////////////////////// computing BR P1 ///////////////////////////////////////////
		
		std::shared_ptr<VectorMG> vecLob = std::dynamic_pointer_cast<partialQValueFunction>(lb)->getSupportVector(0);
		//std::cout << "\n vec : " << vecLob->str();
		
		std::shared_ptr<StochasticDecisionRule> extractedStrategyP1 = algo->extractStrategy(vecLob,horizon,1);
		
		//std::cout << "\n extracted strategy player 2 : " << extractedStrategy->str();
		std::cout << "\n extracted strategy player 1 : " << extractedStrategyP1->str();
		//std::exit(1);

		int solveBRP1 = 1;
		if (solveBRP1){
			//now solve the BR 
			// Parse file into MPOMDP

			const number num_player_ = 1;
			// Instanciate the problem
			std::shared_ptr<SolvableByHSVI> hsvi_mdp_br = std::make_shared<PrivateOccupancyMDP>(mdp, extractedStrategyP1, num_player_,-1, true, true);
			
			/*
			std::cout << "initial state : " << hsvi_mdp->initial_state_->str()<<std::flush;
			std::cout << "\naction space :  " << *mdp->getActionSpaceAt(0,0)<<std::endl;
			std::cout << "\ndr action space : " << *hsvi_mdp->getActionSpaceAt(hsvi_mdp->getInitialState(),0);
			*/

			// Instanciate Initializer
			auto lb_init_br = std::make_shared<MinInitializer>(hsvi_mdp_br);
			auto ub_init_br = std::make_shared<MaxInitializer>(hsvi_mdp_br);

			// Instanciate action selection 
			auto action_tabular_br = std::make_shared<ExhaustiveActionSelection>(hsvi_mdp_br);

			// Declare bounds
			std::shared_ptr<ValueFunction> lb_br, ub_br;

			// Instanciate lower bound
			lb_br = std::make_shared<TabularValueFunction>(hsvi_mdp_br, lb_init_br, action_tabular_br);

			// Instanciate lower bound update operator
			auto lb_update_operator_br = std::make_shared<TabularUpdate>(lb_br);
			lb_br->setUpdateOperator(lb_update_operator_br);

			// Instanciate upper bound
			ub_br = std::make_shared<TabularValueFunction>(hsvi_mdp_br, ub_init_br, action_tabular_br);

			// Instanciate upper bound update operator
			auto ub_update_operator_br = std::make_shared<TabularUpdate>(ub_br);
			ub_br->setUpdateOperator(ub_update_operator_br);
			
			// Instanciate HSVI
			auto algo_br = std::make_shared<HSVI>(hsvi_mdp_br, lb_br, ub_br, error, 10000, "", 1, 1);
			algo_br->agent_id_ = (num_player_*2)-1;
			
			//auto algo = std::make_shared<ValueIteration>(hsvi_mdp,lb,error,100000,"");

			// Initialize and solve the problem
			algo_br->initialize();
			algo_br->solve();
		}
	}
	catch (exception::Exception &e)
	{
		std::cout << "!!! Exception: " << e.what() << std::endl;
	}

	return 0;
} // END main