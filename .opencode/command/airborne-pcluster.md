---
description: Connect to interactive compute node on airborne pcluster for testing
---

Run the following in a background SESSION:

`ssh airborne_pcluster` to connect to the head node of the AWS ParallelCluster

Use `sinfo` to generate a list of partitions.

Based on the list from `sinfo`, ask the user which partition to use for the compute node (default: same as the system default).

`srun --pty bash` to connect to an interactive compute node. If a partition is provided, connect to that partition.

`cd join_scratch` to enter a copy of the current project on the remote. Run `git status` to figure out where you are.

Now, ask the user how to proceed next:

1. Work in the current directory (default). In this case, work in the `join_scratch` directory. Check out the current *local* branch on the remote. Confirm that the local and remote versions are on the same commit. When making local changes, commit and push from the local machine; then, on the remote, pull in the changes.

2. Work in a worktree. In this case, create a git worktree in detached head state with a path like `../join_scratch-worktree-YYYYMMDD_HHMM` (replacing the YYYYMMDD_HHMM with the current timestamp). Point the detached head to the same commit as the local branch (via the branch reference -- `origin/<branch name>`). When working like this, do the same local workflow as above --- commit + push. On the remote, instead of pull, run `git checkout --detach origin/<branch-name>` again to get the latest code.

3. Something else -- in this case, prompt for user input.

Once the workflow is set up, run tests, benchmarks, etc. develop code locally but run tests in the remote process.
