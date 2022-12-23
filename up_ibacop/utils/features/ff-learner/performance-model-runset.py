#!/usr/bin/python

#Runing the computation of init-state features over a probset (with prefix)
#for generating the per-instance performance model.

import sys, os, glob

base_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
domain_file = os.path.realpath(sys.argv[1])
probset_dir = os.path.realpath(sys.argv[2])
probs_prefix = sys.argv[3]
knowledge_path = os.path.realpath(sys.argv[4]) + "/"

print probset_dir, probs_prefix, knowledge_path



planner_args = " -S 28 -k " + knowledge_path

for probfile in sorted(glob.glob(probset_dir + "/" + probs_prefix + "*")):

    call = base_dir + "/roller3.0 -o " + domain_file + " -f " + probfile + planner_args
    print call
    os.system(call)
