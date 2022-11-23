#include <stdexcept>
#include <cstdlib>
#include <iostream>

#include <sdm/config.hpp>
#include <sdm/exception.hpp>
#include <sdm/parser/parser.hpp>

#include <sdm/world/solvable_by_hsvi.hpp>
#include <sdm/world/occupancy_mdp.hpp>
#include <sdm/world/private_occupancy_mdp.hpp>
#include <sdm/world/occupancy_mg.hpp>
#include <sdm/algorithms/planning/hsvi.hpp>
#include <sdm/algorithms/planning/dp.hpp>
#include <sdm/algorithms/planning/value_iteration.hpp>
#include <sdm/utils/value_function/action_selection/exhaustive_action_selection.hpp>
#include <sdm/utils/value_function/update_operator/vupdate/tabular_update.hpp>
#include <sdm/utils/value_function/vfunction/tabular_value_function.hpp>
#include <sdm/utils/linear_programming/lp_last_step_posg.hpp>


using namespace sdm;


    std::shared_ptr<JointHistoryInterface> getRevertedHistory(std::shared_ptr<HistoryInterface> h1, std::shared_ptr<HistoryInterface> h2, const std::shared_ptr<OccupancyStateMG>& occupancy_state, int agent_id_)
    {
        //std::cout << "\n history opponent : " << h1->short_str() << " history : " << h2->short_str();
        if (agent_id_==1){

            for (auto& jh : occupancy_state->getJointHistories()){

                if (jh->getIndividualHistory(1)->short_str() == h1->short_str() && jh->getIndividualHistory(0)->short_str()==h2->short_str()){
                    return jh;
                }
            
            }
        }
        else{
            //std::cout << "\n hist : " << hist->str();
            for (auto& jh : occupancy_state->getJointHistories()){
                
                if (jh->getIndividualHistory(0)->short_str() == h1->short_str() && jh->getIndividualHistory(1)->short_str()==h2->short_str()){
                    return jh;
                }
            }
        }
        //std::cout << "returning null ptr when getting reverted history";
        //std::exit(1);
        return nullptr;
    }


    //helper functions
    std::shared_ptr<sdm::JointAction> getActionPointer(std::shared_ptr<sdm::Action> a, std::shared_ptr<sdm::Action> b,
    std::shared_ptr<POMDPInterface> pomdp, number agent_id_)
    {
        if (agent_id_==0){
            for (const auto &joint_action : *pomdp->getActionSpace()){
                if (joint_action->toAction()->toJointAction()->get(0)->str()==a->str()
                && joint_action->toAction()->toJointAction()->get(1)->str()==b->str()){
                    return joint_action->toAction()->toJointAction();
                }
            }
        }
        if (agent_id_==1){
            for (const auto &joint_action : *pomdp->getActionSpace()){
                if (joint_action->toAction()->toJointAction()->get(0)->str()==b->str()
                && joint_action->toAction()->toJointAction()->get(1)->str()==a->str()){
                    return joint_action->toAction()->toJointAction();
                }
            }
        }
        std::cout << "\n i'm returning null ptr when getting actions";
        std::exit(1);
        return nullptr;
    }

double testValues(std::shared_ptr<StochasticDecisionRule>& strategy, std::shared_ptr<OccupancyStateMG>& state, std::shared_ptr<SolvableByHSVI>& world, int agent_id_){
	if (agent_id_==0){
		double res = 0.0;
		std::shared_ptr<OccupancyMG> world_ = std::dynamic_pointer_cast<OccupancyMG>(world);
		for (auto & history_opponent : state->getIndividualHistories((agent_id_+1)%2)){
			
			double value=INFINITY;

			for (auto & action_opponent : * world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((agent_id_+1)%2)){

				double val=0.0;

				for (auto & hist : state->getIndividualHistories(agent_id_)){
					for (auto & action : * world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(agent_id_)){
											
						std::shared_ptr<Action> jact = getActionPointer(action->toAction(),action_opponent->toAction(),world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),agent_id_)->toAction();

						auto  joint_history_reverted = getRevertedHistory(hist,history_opponent,state, agent_id_);
						if (joint_history_reverted!=0){
							val+= state->getBeliefAt(joint_history_reverted)->getReward(world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)*state->getProbability(joint_history_reverted)*strategy->getProbability(hist,action->toAction());
						}
					}
				}
				//std::cout << "\n value : " << val << "for action : " << action_opponent->str();
				if (val<value){
					value=val;
				}

			}
			res+=value;
		}	
		return res;

	}
	else{
		double res = 0.0;
		std::shared_ptr<OccupancyMG> world_ = std::dynamic_pointer_cast<OccupancyMG>(world);
		for (auto & history_opponent : state->getIndividualHistories((agent_id_+1)%2)){
			
			double value=-INFINITY;

			for (auto & action_opponent : * world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace((agent_id_+1)%2)){

				double val=0.0;

				for (auto & hist : state->getIndividualHistories(agent_id_)){
					for (auto & action : * world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP()->getActionSpace()->toMultiDiscreteSpace()->getSpace(agent_id_)){
											
						std::shared_ptr<Action> jact = getActionPointer(action->toAction(),action_opponent->toAction(),world_ ->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(),agent_id_)->toAction();

						auto  joint_history_reverted = getRevertedHistory(hist,history_opponent,state, agent_id_);
						if (joint_history_reverted !=0){
						val+= state->getBeliefAt(joint_history_reverted)->getReward(world_->getUnderlyingBeliefMDP()->getUnderlyingPOMDP(), jact,0)*state->getProbability(joint_history_reverted)*strategy->getProbability(hist,action->toAction());
						}
					}
				}
				//std::cout << "\n value : " << val << "for action : " << action_opponent->str();
				if (val>value){
					value=val;
				}

			}
			res+=value;
		}
		return res;

	}
	//std::cout << "\n best-response value = " << res;
}

int main(int argc, char **argv)
{
	std::vector<std::string> files;
	files.push_back("/home/delage/gitSdms3/sdms/data/world/dpomdp/adversarial_tiger.dpomdp");
	files.push_back("/home/delage/gitSdms3/sdms/data/world/dpomdp/mabc.dpomdp");
	files.push_back("/home/delage/gitSdms3/sdms/data/world/dpomdp/recycling.dpomdp");
	files.push_back("/home/delage/gitSdms3/sdms/data/world/dpomdp/competitive_tiger.dpomdp");
	
	//std::string filename = "/home/delage/gitSdms3/sdms/data/world/dpomdp/adversarial_tiger.dpomdp";
	number horizon = 3, truncation = 10;
	double error = 0.0000000001, discount = 1.;
	try{
		for (std::string & filename : files){
			for (int id = 0;id<2;id++){
			// Parse file into MPOMDP
			//std::cout << "filename : "  << filename <<std::flush;
			auto mdp = sdm::parser::parse_file(filename);
			//std::cout<<"after the parsin" << std::flush;
			mdp->setHorizon(horizon);
			mdp->setDiscount(discount);

			const number num_player_ = 0;
			// Instanciate the problem
			std::shared_ptr<SolvableByHSVI> hsvi_mdp = std::make_shared<OccupancyMG>(mdp, num_player_,(truncation > 0) ? truncation : horizon, true, true);
			//std::cout << "horizon : " << hsvi_mdp->getHorizon() << std::flush;
			std::shared_ptr<Action> dr1 = std::make_shared<StochasticDecisionRule>(3);
			std::shared_ptr<Action> dr2 = std::make_shared<StochasticDecisionRule>(3);
			//Joint jointDr({dr1,dr2});
			 std::vector<std::shared_ptr<Action>> vec = {dr1,dr2};
            std::shared_ptr<JointAction> jointDr = std::make_shared<JointAction>(vec);
			//auto jdr = std::make_shared<JointDeterministicDecisionRule>(sdm::JointDeterministicDecisionRule(jointDr,mdp->getActionSpace()));
			std::shared_ptr<OccupancyStateMG> nextState =  std::dynamic_pointer_cast<OccupancyStateMG>(hsvi_mdp->getNextState(hsvi_mdp->getInitialState(),jointDr->toAction(),sdm::NO_OBSERVATION,0).first);

			std::shared_ptr<OccupancyStateMG> nextStateBis =  std::dynamic_pointer_cast<OccupancyStateMG>(hsvi_mdp->getNextState(nextState,jointDr->toAction(),sdm::NO_OBSERVATION,1).first);

			//std::cout << nextState->str();
			//std::exit(1);
			std::shared_ptr<LPLastStepPOSG> lp = std::make_shared<LPLastStepPOSG>(hsvi_mdp,id);
			//std::cout << "\n end";
			auto res = lp->createLP(nextStateBis,0);
			
			std::cout << "\n strategy returned : " << res.first->str();
			double valLP = 0.0;
			for (std::map<std::shared_ptr<HistoryInterface>,double>::iterator it=res.second.begin(); it !=res.second.end();++it){
				valLP+=it->second*nextStateBis->getProbabilityOverIndividualHistories((id+1)%2, it->first);
			}
			double valBR = testValues(res.first,nextStateBis,hsvi_mdp,id);
			std::cout << "\n valBR : " << valBR;
			std::cout << "\n------------------\n <comparing LP value and BR value>";
			if (valBR<valLP-0.0001 || valBR>valLP+0.001){
				std::cout << "\n test failed";
				throw std::invalid_argument( "test lp value and br value failed" );
			}
			std::cout << "\n <test passed>";
			//std::cout << "\n end  test";
			}
			//std::exit(1);
		}
		std::exit(1);
	}
	catch (exception::Exception &e)
	{
		std::cout << "!!! Exception: " << e.what() << std::endl;
	}

	return 0;
} // END main
