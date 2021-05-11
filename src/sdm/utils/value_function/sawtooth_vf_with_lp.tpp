
namespace sdm
{

    template <typename TState, typename TAction, typename TValue>
    SawtoothValueFunctionLP<TState, TAction, TValue>::SawtoothValueFunctionLP() {}

    template <typename TState, typename TAction, typename TValue>
    SawtoothValueFunctionLP<TState, TAction, TValue>::SawtoothValueFunctionLP(std::shared_ptr<SolvableByHSVI<TState, TAction>> problem, number horizon, std::shared_ptr<Initializer<TState, TAction>> initializer) : DecentralizedConstraintsLP<TState, TAction, TValue>(problem),SawtoothValueFunction<TState, TAction, TValue>(problem, horizon, initializer)
    {
    }

    template <typename TState, typename TAction, typename TValue>
    SawtoothValueFunctionLP<TState, TAction, TValue>::SawtoothValueFunctionLP(std::shared_ptr<SolvableByHSVI<TState, TAction>> problem, number horizon, TValue default_value) : SawtoothValueFunctionLP<TState, TAction, TValue>(problem, horizon, std::make_shared<ValueInitializer<TState, TAction>>(default_value))
    {
    }

    template <typename TState, typename TAction, typename TValue>
    TAction SawtoothValueFunctionLP<TState, TAction, TValue>::getBestAction(const TState & compressed_occupancy_state, number t)
    {
        double clb = 0, cub = 0;
        return this->greedySawtooth(compressed_occupancy_state, clb, cub, t);
    }

    template <typename TState, typename TAction, typename TValue>
    TAction SawtoothValueFunctionLP<TState, TAction, TValue>::greedySawtooth(const TState& occupancy_state, double clb, double& cub, number t)
    {
        number c = 0;

        //<! tracking variables
        std::string VarName;

        TAction a;

        std::unordered_map<agent, std::unordered_set<typename TState::jhistory_type::element_type::ihistory_type>> ihs;

        for(number ag=0; ag<this->getWorld()->getUnderlyingProblem()->getNumAgents(); ++ag)
        {
            std::unordered_set<typename TState::jhistory_type::element_type::ihistory_type> empty;
            ihs.emplace(ag, empty);
        }

        IloEnv env;
        try{
            IloModel model(env);

            // Init the model
            IloRangeArray con(env);
            IloNumVarArray var(env);

            IloObjective obj = IloMaximize(env);

            ///////  BEGIN CORE CPLEX Code  ///////

            // 0. Build variables a(u|o), a_i(u_i|o_i), v
            this->setGreedyVariables(occupancy_state, ihs, env, var, clb, cub, t);

            // 1. Build objective function \sum_{o,u} a(u|o) \sum_x s(x,o) Q_MDP(x,u) - discount * v 
            this->setGreedyObjective(occupancy_state, obj, var,t);

            //<! 3.a Build sawtooth constraints v <= (V_k - v_k) * \frac{\sum_{o,u} a(u|o)\sum_{x,z_} s(x,o)*p(x,u,z_,x_)}}{s_k(x_,o_)} ,\forall k, x_,o_
            if(!this->representation[t+1].empty())  this->template setGreedySawtooth<TState>(occupancy_state,model, env, con, var, c, t);

            // 3. Build decentralized control constraints [  a(u|o) >= \sum_i a_i(u_i|o_i) + 1 - n ] ---- and ---- [ a(u|o) <= a_i(u_i|o_i) ]
            this->template setDecentralizedConstraints<TState>(occupancy_state, ihs, env, con, var, c, t);

            ///////  END CORE  CPLEX Code ///////
            model.add(obj);
            model.add(con);
            IloCplex cplex(model);
            cplex.setOut(env.getNullStream());
            cplex.setWarning(env.getNullStream());

            // Optimize the problem and obtain solution
            cplex.exportModel("bellman_greedy_op.lp");
            if( !cplex.solve() )
            {
                env.error() << "Failed to optimize MILP" << std::endl;
                system("cat bellman_greedy_op.lp");
                throw(-1);
            }
            else
            {
                cub = cplex.getObjValue();
                a = this->template getDecentralizedVariables<TState>(cplex, var, occupancy_state);
                if( std::abs(cub - this->getQValueAt(occupancy_state, a, t)) > 0.01 )
                {
                    throw sdm::exception::Exception("Unexpected upper-bound values : cub(" + std::to_string(cub) + ")\t vub(" + std::to_string(this->getQValueAt(occupancy_state, a, t)) +  ")");
                }
            }
        }catch(IloException& e)
        {
            std::cerr << "Concert exception caught: " << e << std::endl;
            exit(-1);
        }catch (const std::exception &exc)
        {
            // catch anything thrown within try block that derives from std::exception
            std::cerr << "Non-Concert exception caught: " << exc.what() << std::endl;
            exit(-1);
        }

        env.end();

        return a;
    }

    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<OccupancyState<>, T>, int>>
    void SawtoothValueFunctionLP<TState, TAction, TValue>::testFunction(const TState& occupancy_state, TAction det_action, number t)
    {
        for(const auto compressed_occupancy_state_AND_upper_bound : this->representation[t+1])
        {
            auto upper_bound = compressed_occupancy_state_AND_upper_bound.second;
            auto compressed_occupancy_state = compressed_occupancy_state_AND_upper_bound.first;
            auto initial_upper_bound = this->getInitFunction()->operator()(compressed_occupancy_state, t+1);
            auto next_one_step_uncompressed_occupancy_state = compressed_occupancy_state.getOneStepUncompressedOccupancy();
            auto difference = upper_bound - initial_upper_bound; 

            // Go over all joint histories in over the support of next_one_step_uncompressed_occupancy_state
            for(const auto &pair_hidden_state_AND_joint_history_AND_probability : *next_one_step_uncompressed_occupancy_state)
            {
                auto next_hidden_state = pair_hidden_state_AND_joint_history_AND_probability.first.first;
                auto next_joint_history = pair_hidden_state_AND_joint_history_AND_probability.first.second;
                
                auto joint_history = next_joint_history->getParent();
                auto next_observation = next_joint_history->getData();

                if(occupancy_state.getJointHistories().find(joint_history) != occupancy_state.getJointHistories().end())
                {
                    auto action = det_action.act(joint_history->getIndividualHistories());

                    std::cout<<"\n action "<<action<<std::endl;

                    std::cout<<"\n compressed_occupancy_state.getOneStepUncompressedOccupancy()"<<compressed_occupancy_state.getOneStepUncompressedOccupancy()<<std::endl;
                    std::cout<<"\n joint_history "<<joint_history<<std::endl;
                    std::cout<<"\n next_hidden_state "<<next_hidden_state<<std::endl;
                    std::cout<<"\n next_observation "<<next_observation<<std::endl;
                    std::cout<<"\n next_one_step_uncompressed_occupancy_state"<<next_one_step_uncompressed_occupancy_state<<std::endl;
                    
                    auto factor = 0.0;

                    for(const auto& hidden_state : next_one_step_uncompressed_occupancy_state->getStatesAt(joint_history))
                    {
                        factor += next_one_step_uncompressed_occupancy_state->at(std::make_pair(hidden_state, joint_history)) * this->getWorld()->getUnderlyingProblem()->getObsDynamics()->getDynamics(hidden_state, this->getWorld()->getUnderlyingProblem()->getActionSpace()->joint2single(action), this->getWorld()->getUnderlyingProblem()->getObsSpace()->joint2single(next_observation), next_hidden_state);
                    }
                    auto denominator = next_one_step_uncompressed_occupancy_state->at(std::make_pair(next_hidden_state, joint_history->expand(next_observation)));
                    auto resultat =  difference * factor / denominator;                
                    
                    std::cout<<"\n factor "<<factor<<", denominator "<<denominator<<", difference "<<difference<<", resultat "<<resultat<<std::endl;
                }
            }
        }
    }

    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<SerializedOccupancyState<>, T>, int>>
    void SawtoothValueFunctionLP<TState, TAction, TValue>::testFunction(const TState& , TAction , number )
    {}

    template <typename TState, typename TAction, typename TValue>
    void SawtoothValueFunctionLP<TState, TAction, TValue>::setGreedyVariables(const TState& occupancy_state, std::unordered_map<agent, std::unordered_set<typename TState::jhistory_type::element_type::ihistory_type>>& ihs, IloEnv& env, IloNumVarArray& var, double /*clb*/, double /*cub*/, number t)
    {
        //<! tracking variable ids
        number index = 0;

        //<! tracking variables
        std::string VarName;

        this->variables.clear();

        //<! 0.b Build variables v_0 = objective variable!
        VarName = this->getVarNameWeight(0);
        var.add(IloNumVar(env, -IloInfinity, 0.0, VarName.c_str())); 
        this->setNumber(VarName, index++);

        //<! Define variables \omega_k(x',o')

        // Go over all Point Set in t+1 
        for(auto compressed_occupancy_state_AND_upper_bound : this->representation[t+1])
        {
            auto compressed_occupancy_state = compressed_occupancy_state_AND_upper_bound.first;
            auto one_step_uncompressed_occupancy_state = compressed_occupancy_state.getOneStepUncompressedOccupancy();
           
            // Go over all Joint History Next
            for(const auto& hidden_state_AND_joint_history_AND_probability : *one_step_uncompressed_occupancy_state)
            {
                auto hidden_state_AND_joint_history = hidden_state_AND_joint_history_AND_probability.first;
                
                auto hidden_state = one_step_uncompressed_occupancy_state->getState(hidden_state_AND_joint_history);
                auto joint_history = one_step_uncompressed_occupancy_state->getHistory(hidden_state_AND_joint_history);

                // <! \omega_k(x',o')
                VarName = this->getVarNameWeightedStateJointHistory(one_step_uncompressed_occupancy_state, hidden_state, joint_history);
                var.add(IloBoolVar(env, 0, 1, VarName.c_str()));
                this->setNumber(VarName, index++);
           }
        }
        this->template setDecentralizedVariables<TState>(occupancy_state, ihs, env, var, index, t);
    }

    template <typename TState, typename TAction, typename TValue>
    void SawtoothValueFunctionLP<TState, TAction, TValue>::setGreedyObjective(const TState& compressed_occupancy_state, IloObjective& obj, IloNumVarArray& var,number t) 
    {
        // <! 1.a get variable v
        auto recover = this->getNumber(this->getVarNameWeight(0));

        //<! 1.b set coefficient of objective function "\sum_{o,u} a(u|o) \sum_x s(x,o) Q_MDP(x,u) - discount * v0"
        obj.setLinearCoef(var[recover], this->getWorld()->getUnderlyingProblem()->getDiscount(t));

        // Go over all joint history
        for(const auto &joint_history : compressed_occupancy_state.getJointHistories())
        {
            // Go over all action 
            for(const auto & action : this->getWorld()->getUnderlyingProblem()->getActionSpace()->getAll())
            {
                //<! 1.c.4 get variable a(u|o)
                recover = this->getNumber(this->getVarNameJointHistoryDecisionRule(action, joint_history));

                //<! 1.c.5 set coefficient of variable a(u|o) i.e., \sum_x s(x,o) Q_MDP(x,u)
                obj.setLinearCoef(var[recover], this->template getQValueRelaxation<TState>(compressed_occupancy_state, joint_history, action, t));
            }
        }
    }

    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<OccupancyState<>, T>, int>>
    double SawtoothValueFunctionLP<TState, TAction, TValue>::getQValueRelaxation(const TState& compressed_occupancy_state,typename TState::jhistory_type joint_history, typename TAction::output_type action, number t) 
    {
        auto weight = 0.0;
        for(auto x : this->getWorld()->getUnderlyingProblem()->getStateSpace()->getAll())
        {
            weight +=  compressed_occupancy_state.at(std::make_pair(x,joint_history)) * std::static_pointer_cast<State2OccupancyValueFunction<typename TState::state_type,TState>>(this->getInitFunction())->getQValueAt(x,this->getWorld()->getUnderlyingProblem()->getActionSpace()->joint2single(action),t);
        }

        return weight;
    }

    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<SerializedOccupancyState<>, T>, int>>
    double SawtoothValueFunctionLP<TState, TAction, TValue>::getQValueRelaxation(const TState& ,typename TState::jhistory_type , typename TAction::output_type , number ) 
    {
        return 0;
    }

    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<OccupancyState<>, T>, int>>
    double SawtoothValueFunctionLP<TState, TAction, TValue>::getQValueRealistic(const TState& compressed_occupancy_state, typename TState::jhistory_type joint_history, typename TAction::output_type action, typename TState::state_type next_hidden_state, typename TState::observation_type next_observation, double denominator, double difference)
    {
        return difference * this->template getSawtoothMinimumRatio<TState>(*compressed_occupancy_state.getOneStepUncompressedOccupancy(), joint_history, action, next_hidden_state, next_observation, denominator);
    }

    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<SerializedOccupancyState<>, T>, int>>
    double SawtoothValueFunctionLP<TState, TAction, TValue>::getQValueRealistic(const TState&, typename TState::jhistory_type, typename TAction::output_type, typename TState::state_type, typename TState::observation_type, double, double)
    {
        return 0;
    }


    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<OccupancyState<>, T>, int>>
    void SawtoothValueFunctionLP<TState, TAction, TValue>::setGreedySawtooth(const TState& occupancy_state, IloModel& model, IloEnv& env, IloRangeArray& con, IloNumVarArray& var, number& c, number t) 
    {
        //<!  Build sawtooth constraints v - \sum_{u} a(u|o) * Q(k,s,o,u,y,z, diff, t  ) + \omega_k(y,<o,z>)*M <= M,  \forall k, y,<o,z>
        //<!  Build sawtooth constraints  Q(k,s,o,u,y,z, diff, t ) = (v_k - V_k) \frac{\sum_{x} s(x,o) * p(x,u,z,y)}}{s_k(y,<o,z>)},  \forall a(u|o)

        assert(this->getInitFunction() != nullptr); 
        number recover = 0;
        number bigM = 1;

        // Go over all points in the point set at t+1 
        for(const auto compressed_occupancy_state_AND_upper_bound : this->representation[t+1])
        {
            auto current_upper_bound = compressed_occupancy_state_AND_upper_bound.second;
            auto compressed_occupancy_state = compressed_occupancy_state_AND_upper_bound.first;
            auto initial_upper_bound = this->getInitFunction()->operator()(compressed_occupancy_state, t+1);
            auto next_one_step_uncompressed_occupancy_state = compressed_occupancy_state.getOneStepUncompressedOccupancy();
            auto difference = current_upper_bound - initial_upper_bound; 

            // Go over all joint histories in over the support of next_one_step_uncompressed_occupancy_state
            for(const auto &pair_hidden_state_AND_joint_history_AND_probability : *next_one_step_uncompressed_occupancy_state)
            {
                con.add(IloRange(env, -IloInfinity, bigM));
                con[c].setLinearCoef(var[this->getNumber(this->getVarNameWeight(0))], +1.0);

                auto probability = pair_hidden_state_AND_joint_history_AND_probability.second;
                auto next_hidden_state = pair_hidden_state_AND_joint_history_AND_probability.first.first;
                auto next_joint_history = pair_hidden_state_AND_joint_history_AND_probability.first.second;
                
                auto joint_history = next_joint_history->getParent();
                auto next_observation = next_joint_history->getData();

                if(occupancy_state.getJointHistories().find(joint_history) != occupancy_state.getJointHistories().end())
                {
                    // Go over all actions
                    for(const auto & action : this->getWorld()->getUnderlyingProblem()->getActionSpace()->getAll())
                    {
                        //<! 1.c.4 get variable a(u|o) and set constant 
                        con[c].setLinearCoef(var[this->getNumber(this->getVarNameJointHistoryDecisionRule(action, joint_history))], - this->template getQValueRealistic<TState>(occupancy_state, joint_history, action, next_hidden_state, next_observation, probability, difference) );
                    } 
                } 

                con[c].setLinearCoef(var[this->getNumber(this->getVarNameWeightedStateJointHistory(next_one_step_uncompressed_occupancy_state, next_hidden_state, next_joint_history))], bigM);

                c++;
                
                /*
                //<! Initialize expression
                IloExpr expr(env);

                //<! 1.c.1 get variable v and set coefficient of variable v
                expr = var[this->getNumber(this->getVarNameWeight(0))];

                auto next_hidden_state = pair_hidden_state_AND_joint_history_AND_probability.first.first;
                auto next_joint_history = pair_hidden_state_AND_joint_history_AND_probability.first.second;
                
                auto joint_history = next_joint_history->getParent();
                auto next_observation = next_joint_history->getData();

                if(occupancy_state.getJointHistories().find(joint_history) != occupancy_state.getJointHistories().end())
                {
                    // Go over all actions
                    for(const auto & action : this->getWorld()->getUnderlyingProblem()->getActionSpace()->getAll())
                    {
                        //<! 1.c.4 get variable a(u|o) and set constant 
                        expr -= this->template getQValueRealistic<TState>(occupancy_state, joint_history, action, next_hidden_state, next_observation, probability, difference) * var[this->getNumber(this->getVarNameJointHistoryDecisionRule(action, joint_history))];
                    } 
                }               

                // <! get variable \omega_k(x',o')
                recover = this->getNumber(this->getVarNameWeightedStateJointHistory(next_one_step_uncompressed_occupancy_state, next_hidden_state, next_joint_history));
                model.add( IloIfThen(env, var[recover] > 0, expr <= 0) );
                */
            }

            // Build constraint \sum{x',o'} \omega_k(x',o') = 1
            con.add(IloRange(env, 1.0, 1.0));
            for(const auto &pair_hidden_state_AND_joint_history_AND_probability : *next_one_step_uncompressed_occupancy_state)
            {
                auto next_hidden_state = pair_hidden_state_AND_joint_history_AND_probability.first.first;
                auto next_joint_history = pair_hidden_state_AND_joint_history_AND_probability.first.second;

                // <! \omega_k(x',o')
                recover = this->getNumber(this->getVarNameWeightedStateJointHistory(next_one_step_uncompressed_occupancy_state, next_hidden_state, next_joint_history));
                con[c].setLinearCoef(var[recover], +1.0);
            }
            c++;
        }
    }


    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<OccupancyState<>, T>, int>>
    double SawtoothValueFunctionLP<TState, TAction, TValue>::getSawtoothMinimumRatio(const TState& one_step_uncompressed_occupancy_state, typename TState::jhistory_type joint_history, typename TAction::output_type action, typename TState::state_type next_hidden_state, typename TState::observation_type next_observation, double denominator)
    {
        auto factor = 0.0;

        for(const auto& hidden_state : one_step_uncompressed_occupancy_state.getStatesAt(joint_history))
        {
            factor += one_step_uncompressed_occupancy_state.at(std::make_pair(hidden_state, joint_history)) * this->getWorld()->getUnderlyingProblem()->getObsDynamics()->getDynamics(hidden_state, this->getWorld()->getUnderlyingProblem()->getActionSpace()->joint2single(action), this->getWorld()->getUnderlyingProblem()->getObsSpace()->joint2single(next_observation), next_hidden_state);
        }

        return factor / denominator;
    }

    template <typename TState, typename TAction, typename TValue>
    void SawtoothValueFunctionLP<TState, TAction, TValue>::updateValueAt(const TState &occupancy_state, number t)
    {
        double cub = 0, clb = 0;

        this->greedySawtooth(occupancy_state, clb, cub, t);

        MappedValueFunction<TState,TAction,TValue>::updateValueAt(occupancy_state, t, cub);
    }


    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<SerializedOccupancyState<>, T>, int>>
    void SawtoothValueFunctionLP<TState, TAction, TValue>::setGreedySawtooth(const TState& serial_occupancy_state, IloModel& model, IloEnv& env, IloRangeArray& con, IloNumVarArray& var, number& c, number t) 
    {

        assert(this->getInitFunction() != nullptr); 

        number recover = 0;
        number agent_id = serial_occupancy_state.getCurrentAgentId();  

        // Go over all points in the point set at t+1 
        for(const auto compressed_occupancy_state_AND_upper_bound : this->representation[t+1])
        {

            auto upper_bound = compressed_occupancy_state_AND_upper_bound.second;
            auto compressed_occupancy_state = compressed_occupancy_state_AND_upper_bound.first;
            auto initial_upper_bound = this->getInitFunction()->operator()(compressed_occupancy_state, t+1);
            auto next_one_step_uncompressed_occupancy_state = compressed_occupancy_state.getOneStepUncompressedOccupancy();
            auto difference = upper_bound - initial_upper_bound; 

            // Go over all joint histories in over the support of next_one_step_uncompressed_occupancy_state
            for(const auto &pair_hidden_serial_state_AND_joint_history_AND_probability : *next_one_step_uncompressed_occupancy_state)
            {
                //<! Initialize expression
                IloExpr expr(env);

                //<! 1.c.1 get variable v and set coefficient of variable v
                expr = var[this->getNumber(this->getVarNameWeight(0))];

                auto pair_hidden_serial_state_AND_joint_history = pair_hidden_serial_state_AND_joint_history_AND_probability.first;

                auto next_hidden_state = next_one_step_uncompressed_occupancy_state->getState(pair_hidden_serial_state_AND_joint_history);
                auto next_joint_history = next_one_step_uncompressed_occupancy_state->getHistory(pair_hidden_serial_state_AND_joint_history);
                
                auto indiv_history = next_joint_history->getIndividualHistory(agent_id);

                auto joint_history = next_joint_history;
                auto next_observation = next_joint_history->getDefaultObs();

                if(agent_id == 0)
                {
                    joint_history = next_joint_history->getParent();
                    next_observation = next_joint_history->getData();
                }

                if(serial_occupancy_state.getJointHistories().find(joint_history) != serial_occupancy_state.getJointHistories().end())
                {
                    // Go over all actions
                    for(const auto & serial_action : this->getWorld()->getUnderlyingProblem()->getActionSpace(t)->getAll())
                    {
                        //<! 1.c.4 get variable a(u|o) and set constant 
                        expr -=  this->getQValueRealistic(serial_occupancy_state, joint_history, serial_action, next_hidden_state, next_observation, *next_one_step_uncompressed_occupancy_state, difference, t)  * var[this->getNumber(this->getVarNameIndividualHistoryDecisionRule(serial_action, indiv_history, agent_id))];
                    } 
                }   
                // <! get variable \omega_k(x',o')
                recover = this->getNumber(this->getVarNameWeightedStateJointHistory(next_one_step_uncompressed_occupancy_state, next_hidden_state, next_joint_history));
                model.add(IloIfThen(env, var[recover] > 0, expr <= 0 ) );      
            }

            // Build constraint \sum{x',o'} \omega_k(x',o') = 1
            con.add(IloRange(env, 1.0, 1.0));
            for(const auto &pair_hidden_state_AND_joint_history_AND_probability : *next_one_step_uncompressed_occupancy_state)
            {
                auto pair_hidden_serial_state_AND_joint_history = pair_hidden_state_AND_joint_history_AND_probability.first;
                auto next_hidden_state = next_one_step_uncompressed_occupancy_state->getState(pair_hidden_serial_state_AND_joint_history);
                auto next_joint_history = next_one_step_uncompressed_occupancy_state->getHistory(pair_hidden_serial_state_AND_joint_history);

                // <! \omega_k(x',o')
                recover = this->getNumber(this->getVarNameWeightedStateJointHistory(next_one_step_uncompressed_occupancy_state, next_hidden_state, next_joint_history));
                con[c].setLinearCoef(var[recover], +1.0);
            }
            c++;
        }
    }

    template <typename TState, typename TAction, typename TValue>
    template <typename T, std::enable_if_t<std::is_same_v<SerializedOccupancyState<>, T>, int>>
    double SawtoothValueFunctionLP<TState, TAction, TValue>::getSawtoothMinimumRatio(const TState& , typename TState::jhistory_type , typename TAction::output_type , typename TState::state_type , typename TState::jhistory_type , double)
    {
        return 0;
    }
}
