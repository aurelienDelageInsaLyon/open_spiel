/* This file is part of the Multiagent Decision Process (MADP) Toolbox. 
 *
 * The majority of MADP is free software released under GNUP GPL v.3. However,
 * some of the included libraries are released under a different license. For 
 * more information, see the included COPYING file. For other information, 
 * please refer to the included README file.
 *
 * This file has been written and/or modified by the following people:
 *
 * Frans Oliehoek 
 * Matthijs Spaan 
 *
 * For contact information please see the included AUTHORS file.
 */

#include "PlanningUnitDecPOMDPDiscrete.h"
#include "State.h"
#include "Action.h"
#include "Observation.h"
#include "StateDistribution.h"
#include <fstream>

using namespace std;

//Default constructor
PlanningUnitDecPOMDPDiscrete::PlanningUnitDecPOMDPDiscrete(
    size_t horizon,
    DecPOMDPDiscreteInterface* p,
    const PlanningUnitMADPDiscreteParameters* params
    ) :
    PlanningUnitMADPDiscrete(horizon,p, params)
    ,_m_DecPOMDP(p)
{
    if(DEBUG_PU_CONSTRUCTORS) cout << "PlanningUnitDecPOMDPDiscrete(PlanningUnitMADPDiscreteParameters params, size_t horizon, DecPOMDPDiscreteInterface* p)  called" << endl;
    if(p!=0)
        SanityCheck();
}
/* 
PlanningUnitDecPOMDPDiscrete::PlanningUnitDecPOMDPDiscrete(
    size_t horizon,
    DecPOMDPDiscreteInterface* p
    ) :
    //Referrer<DecPOMDPDiscreteInterface>(p),
    PlanningUnitMADPDiscrete(horizon,p)
    ,_m_DecPOMDP(p)
{
    if(DEBUG_PU_CONSTRUCTORS) cout << "PlanningUnitDecPOMDPDiscrete(size_t horizon, DecPOMDPDiscreteInterface* p)  called" << endl;
    if(p!=0)
        SanityCheck();
}
 * */

void PlanningUnitDecPOMDPDiscrete::SetProblem(DecPOMDPDiscreteInterface* p)
{
    if(p == _m_DecPOMDP )
        return;
    _m_DecPOMDP = p;

    //set (and initialize) the problem at PlanningUnitMADPDiscrete level:
    MultiAgentDecisionProcessDiscreteInterface* p2 = 
        static_cast<MultiAgentDecisionProcessDiscreteInterface*>(p);
    PlanningUnitMADPDiscrete::SetProblem(p2);

    SanityCheck();
}

bool PlanningUnitDecPOMDPDiscrete::SanityCheck() const
{
    bool sane=PlanningUnitMADPDiscrete::SanityCheck();

    // we cannot check anything check
    if(_m_DecPOMDP==0)
        return(sane);

    if(GetDiscount() < 0 || GetDiscount() > 1)
    {
        sane=false;
        throw(E("PlanningUnitDecPOMDPDiscrete::SanityCheck() failed, discount should be between 0 and 1"));
    }

    if(GetHorizon() == MAXHORIZON &&
       abs(GetDiscount()-1) < PROB_PRECISION)
    {
        sane=false;
        throw(E("PlanningUnitDecPOMDPDiscrete::SanityCheck() failed, in the infinite-horizon case the discount should be less than one"));
    }

    return(sane);
}

void PlanningUnitDecPOMDPDiscrete::ExportDecPOMDPFile(const string & filename) const
{
    ExportDecPOMDPFile(filename,GetDPOMDPD());
}

/// Export is in the dpomdp file format.
void PlanningUnitDecPOMDPDiscrete::ExportDecPOMDPFile(
    const string & filename,
    const DecPOMDPDiscreteInterface *decpomdp)
{
    size_t nrAg=decpomdp->GetNrAgents(),
        nrS=decpomdp->GetNrStates();
    ofstream fp(filename.c_str());
    if(!fp)
    {
        stringstream ss;
        ss << "PlanningUnitDecPOMDPDiscrete::ExportDecPOMDPFile: failed to open file "
           << filename;
        throw(E(ss.str()));
    }

    fp << "agents: " << nrAg << endl;
    fp << "discount: " << decpomdp->GetDiscount() << endl;
    switch(decpomdp->GetRewardType())
    {
    case REWARD:
        fp << "values: reward" << endl;
        break;
    case COST:
        fp << "values: cost" << endl;
    }

    fp << "states:";
    for(Index s=0;s<nrS;s++)
        fp << " "  << decpomdp->GetState(s)->SoftPrintBrief();
    fp << endl;

    StateDistribution* isd = decpomdp->GetISD();
    fp << "start:" << endl;
    for(Index s0=0;s0<nrS;s0++)
    {
        if(s0>0)
            fp << " ";
        double bs = isd->GetProbability(s0);
        fp <<  bs;
    }
    fp << endl;

    fp << "actions:" << endl;
    for(Index i=0;i!=nrAg;++i)
    {
        for(Index a=0;a<decpomdp->GetNrActions(i);a++)
        {
            if(a>0)
                fp << " ";
            fp << decpomdp->GetAction(i,a)->SoftPrintBrief();
        }
        fp << endl;
    }

    fp << "observations:" << endl;
    for(Index i=0;i!=nrAg;++i)
    {
        for(Index o=0;o<decpomdp->GetNrObservations(i);o++)
        {
            if(o>0)
                fp << " ";
            fp << decpomdp->GetObservation(i,o)->SoftPrintBrief();
        }
        fp << endl;
    }

    double p;
    for(Index a=0;a<decpomdp->GetNrJointActions();a++)
        for(Index s0=0;s0<nrS;s0++)
            for(Index s1=0;s1<nrS;s1++)
            {
                p=decpomdp->GetTransitionProbability(s0,a,s1);
                if(p!=0)
                {
                    const vector<Index> &aIs=decpomdp->JointToIndividualActionIndices(a);
                    fp << "T:";
                    for(Index i=0;i!=nrAg;++i)
                        fp << " " << aIs.at(i);
                    fp << " : " << s0 << " : " << s1 << " : " 
                       << p << endl;
                }
            }

    for(Index a=0;a<decpomdp->GetNrJointActions();a++)
        for(Index o=0;o<decpomdp->GetNrJointObservations();o++)
            for(Index s1=0;s1<nrS;s1++)
            {
                p=decpomdp->GetObservationProbability(a,s1,o);
                if(p!=0)
                {
                    const vector<Index> &aIs=decpomdp->JointToIndividualActionIndices(a);
                    const vector<Index> &oIs=decpomdp->JointToIndividualObservationIndices(o);
                    fp << "O:";

                    for(Index i=0;i!=nrAg;++i)
                        fp << " " << aIs.at(i);

                    fp << " : " << s1 << " :";

                    for(Index i=0;i!=nrAg;++i)
                        fp << " " << oIs.at(i);
                    fp << " : " << p << endl;
                }
            }

    for(Index a=0;a<decpomdp->GetNrJointActions();a++)
        for(Index s0=0;s0<nrS;s0++)
        {
            p=decpomdp->GetReward(s0,a);
            if(p!=0)
            {
                const vector<Index> &aIs=decpomdp->JointToIndividualActionIndices(a);
                fp << "R:";

                for(Index i=0;i!=nrAg;++i)
                    fp << " " << aIs.at(i);
                fp << " : " << s0 << " : * : * : " << p << endl;
            }
        }
}
