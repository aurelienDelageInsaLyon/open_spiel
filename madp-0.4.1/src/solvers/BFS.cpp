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

#include <time.h>
#include <sys/times.h>
#include <iostream>
#include <fstream>
#include <float.h>

#include "MADPParser.h"
#include "BruteForceSearchPlanner.h"
#include "directories.h"
#include "DecPOMDPDiscrete.h"
#include "argumentHandlers.h"
#include "argumentUtils.h"

using namespace std;
using namespace ArgumentUtils;


const char *argp_program_version = "BFS";

// Program documentation
static char doc[] =
"BFS - runs the BruteForceSearchPlanner \
\v";

const struct argp_child childVector[] = {
    ArgumentHandlers::problemFile_child,
    ArgumentHandlers::globalOptions_child,
    ArgumentHandlers::outputFileOptions_child,
    ArgumentHandlers::modelOptions_child,
    ArgumentHandlers::solutionMethodOptions_child,
    { 0 }
};

#include "argumentHandlersPostChild.h"

int main(int argc, char **argv)
{
    ArgumentHandlers::Arguments args;
    argp_parse (&ArgumentHandlers::theArgpStruc, argc, argv, 0, 0, &args);
    int restarts = args.nrRestarts;

    try {

    DecPOMDPDiscreteInterface* decpomdp =
        GetDecPOMDPDiscreteInterfaceFromArgs(args);

    stringstream ss;
    ss << directories::MADPGetResultsFilename("BFS",*decpomdp,args)
       << "h" << args.horizon;
    string filename=ss.str();
    ofstream of(filename.c_str());
    if(!of && !args.dryrun)
    {
        cerr << "BFS: could not open " << filename << endl;
        return(1);
    }

    cout << "Computing " << ss.str() << endl;
    
    PlanningUnitMADPDiscreteParameters params;
    params.SetComputeAll(true);
    if(args.sparse)
        params.SetUseSparseJointBeliefs(true);
    else
        params.SetUseSparseJointBeliefs(false);

    BruteForceSearchPlanner bfs(args.horizon, decpomdp, &params);

    cout << "BruteForceSearchPlanner initialized" << endl;


    for(int restartI = 0; restartI < restarts; restartI++)
    {
        tms ts_before, ts_after;
        clock_t ticks_before, ticks_after;
        ticks_before = times(&ts_before);

        bfs.Plan();

        ticks_after = times(&ts_after);
        clock_t ticks =  ticks_after - ticks_before;
        clock_t utime =   ts_after.tms_utime - ts_before.tms_utime;


        double V = bfs.GetExpectedReward();
        cout << "\nvalue="<< V << endl;
//        bfs.GetJointPolicyPureVector()->Print();

        if(! args.dryrun)
        {
            of << args.horizon<<"\t";
            char s[10];
            sprintf(s, "%.6f", V);
            of << s <<"\t";
            of << ticks <<"\t";
            of << utime <<"\t";
            of << bfs.GetJointPolicyPureVector()->GetIndex() <<"\n";
            of.flush();
        }
    }
    }
    catch(E& e){ e.Print(); }
}
