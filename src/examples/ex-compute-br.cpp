
#include <cstdlib>
#include <iostream>

#include <sdm/config.hpp>
#include <sdm/exception.hpp>
#include <sdm/parser/parser.hpp>

#include <sdm/world/solvable_by_hsvi.hpp>
#include <sdm/world/occupancy_mdp.hpp>
#include <sdm/world/private_occupancy_mdp.hpp>
#include <sdm/algorithms/planning/hsvi.hpp>
#include <sdm/algorithms/planning/dp.hpp>
#include <sdm/algorithms/planning/value_iteration.hpp>
#include <sdm/utils/value_function/action_selection/exhaustive_action_selection.hpp>
#include <sdm/utils/value_function/update_operator/vupdate/tabular_update.hpp>
#include <sdm/utils/value_function/vfunction/tabular_value_function.hpp>

using namespace sdm;

int main(int argc, char **argv)
{
	std::string filename = "/home/delage/gitSdms3/sdms/data/world/dpomdp/mabc.dpomdp";
	number horizon = 3, truncation = 0;
	double error = 0.0, discount = 1.;
	try
	{
		// Parse file into MPOMDP
        std::cout << "filename : "  << filename <<std::flush;
		auto mdp = sdm::parser::parse_file(filename);
        std::cout<<"after the parsin" << std::flush;
		mdp->setHorizon(horizon);
		mdp->setDiscount(discount);

		const number num_player_ = 1;
		// Instanciate the problem
		std::shared_ptr<SolvableByHSVI> hsvi_mdp = std::make_shared<PrivateOccupancyMDP>(mdp, num_player_,-1, true, true);
        /*
        std::cout << "initial state : " << hsvi_mdp->initial_state_->str()<<std::flush;
		std::cout << "\naction space :  " << *mdp->getActionSpaceAt(0,0)<<std::endl;
		std::cout << "\ndr action space : " << *hsvi_mdp->getActionSpaceAt(hsvi_mdp->getInitialState(),0);
		*/

		// Instanciate Initializer
		auto lb_init = std::make_shared<MinInitializer>(hsvi_mdp);
		auto ub_init = std::make_shared<MaxInitializer>(hsvi_mdp);

		// Instanciate action selection 
		auto action_tabular = std::make_shared<ExhaustiveActionSelection>(hsvi_mdp);

		// Declare bounds
		std::shared_ptr<ValueFunction> lb, ub;

		// Instanciate lower bound
		lb = std::make_shared<TabularValueFunction>(hsvi_mdp, lb_init, action_tabular);

		// Instanciate lower bound update operator
		auto lb_update_operator = std::make_shared<TabularUpdate>(lb);
		lb->setUpdateOperator(lb_update_operator);

		// Instanciate upper bound
		ub = std::make_shared<TabularValueFunction>(hsvi_mdp, ub_init, action_tabular);

		// Instanciate upper bound update operator
		auto ub_update_operator = std::make_shared<TabularUpdate>(ub);
		ub->setUpdateOperator(ub_update_operator);
		
		// Instanciate HSVI
		auto algo = std::make_shared<HSVI>(hsvi_mdp, lb, ub, error, 10000, "", 1, 1);
		algo->agent_id_ = (num_player_*2)-1;
		
		//auto algo = std::make_shared<ValueIteration>(hsvi_mdp,lb,error,100000,"");

		// Initialize and solve the problem
		algo->initialize();
		algo->solve();
	
	}
	catch (exception::Exception &e)
	{
		std::cout << "!!! Exception: " << e.what() << std::endl;
	}

	return 0;
} // END main