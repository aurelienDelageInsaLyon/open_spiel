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

#include <float.h>
#include <limits.h>
#include "GeneralizedMAAStarPlannerForDecPOMDPDiscrete.h"

#include "QFunctionJAOHInterface.h"

#include "PartialJointPolicyPureVector.h"
#include "PartialJPDPValuePair.h"
#include "PolicyPoolPartialJPolValPair.h"

//#include "JointObservationHistoryTree.h"
//#include "JointBeliefInterface.h"
//#include "JointActionObservationHistoryTree.h"
//#include "BeliefIteratorGeneric.h"


using namespace std;

//Default constructor
GeneralizedMAAStarPlannerForDecPOMDPDiscrete::GeneralizedMAAStarPlannerForDecPOMDPDiscrete(
        size_t horizon, 
        DecPOMDPDiscreteInterface* p
        , const PlanningUnitMADPDiscreteParameters * params
        , int verboseness) 
:
    PlanningUnitDecPOMDPDiscrete(horizon, p, params)
    ,GeneralizedMAAStarPlanner(verboseness)
{
    _m_qHeuristic=0;
    _m_useSparseBeliefs=GetParams().GetUseSparseJointBeliefs();
}
/*
//Destructor
GeneralizedMAAStarPlannerForDecPOMDPDiscrete::
~GeneralizedMAAStarPlannerForDecPOMDPDiscrete()
{
}
*/
//Copy assignment operator
GeneralizedMAAStarPlannerForDecPOMDPDiscrete& GeneralizedMAAStarPlannerForDecPOMDPDiscrete::operator= (const GeneralizedMAAStarPlannerForDecPOMDPDiscrete& o)
{
    if (this == &o) return *this;   // Gracefully handle self assignment
    // Put the normal assignment duties here...
    throw("GeneralizedMAAStarPlannerForDecPOMDPDiscrete::operator= not implemented");

    return *this;
}

//this function extends a previous policy jpolPrevTs for ts-1 with the behavior specified by the policy of the BayesianGame for time step ts (jpolBG).
PJPDP_sharedPtr
GeneralizedMAAStarPlannerForDecPOMDPDiscrete::ConstructExtendedJointPolicy(
        const PartialJointPolicyDiscretePure& jpolPrevTs, 
        const JointPolicyDiscretePure& jpolBG,
        const vector<size_t>& nrOHts, 
        const vector<Index>& firstOHtsI)
{
    //check policy types
    if(jpolPrevTs.GetPolicyDomainCategory() != PolicyGlobals::OHIST_INDEX)
        throw E("GeneralizedMAAStarPlannerForDecPOMDPDiscrete::ConstructExtendedJointPolicy --- jpolPrevTs.GetPolicyDomainCategory() != PolicyGlobals::OHIST_INDEX ");
    if(jpolBG.GetPolicyDomainCategory() != PolicyGlobals::TYPE_INDEX)
        throw E("GeneralizedMAAStarPlannerForDecPOMDPDiscrete::ConstructExtendedJointPolicy --- jpolPrevTs.GetPolicyDomainCategory() != PolicyGlobals::TYPE_INDEX ");
    //construct a policy for the DecPOMDP: 
    //a copy of jpolPrevTs with extended to this time step (ts) by 
    //jpolBG
    PJPDP_sharedPtr jpolTs =
        PJPDP_sharedPtr(
            new PartialJointPolicyPureVector(jpolPrevTs));
    jpolTs->SetDepth( jpolTs->GetDepth()+1 );
    for(Index agentI=0; agentI < GetNrAgents(); agentI++)
    {
        for(Index type = 0; type < nrOHts[agentI]; type++)
        {
            Index ohI = type + firstOHtsI[agentI];
            jpolTs->SetAction(agentI, ohI, 
                    jpolBG.GetActionIndex(agentI, type) );
        }         
    }
    return(jpolTs);
}


PJPDP_sharedPtr GeneralizedMAAStarPlannerForDecPOMDPDiscrete::
NewJPol() const
{
    return PJPDP_sharedPtr(
        new PartialJointPolicyPureVector(this, OHIST_INDEX, 0.0));
}

PartialPolicyPoolItemInterface_sharedPtr
GeneralizedMAAStarPlannerForDecPOMDPDiscrete::
NewPPI(const PJPDP_sharedPtr &jp, double v) const
{
    ///we know that jp is a PartialJointPolicyPureVector (see NewJPol above)
    //this can be used to have a more compact representation by doing:
    //  PartialJointPolicyPureVector* p = 
    //      static_cast<PartialJointPolicyPureVector*>(jp);
    //  ppi = new PartialJPPVIndexValPair(jp,v);

    PartialPolicyPoolItemInterface_sharedPtr ppi =
        PartialPolicyPoolItemInterface_sharedPtr(new PartialJPDPValuePair(jp,v));
    //delete jp;
    return (ppi);
}

PartialPolicyPoolInterface_sharedPtr
GeneralizedMAAStarPlannerForDecPOMDPDiscrete::
NewPP() const
{
    return PartialPolicyPoolInterface_sharedPtr(new PolicyPoolPartialJPolValPair);
}
