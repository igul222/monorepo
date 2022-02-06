Ishaan's machine learning monorepo.

# Organization

- `lib` contains independent modules which are useful across projects.
- `exp` contains one-off throwaway experiments with no maintenance guarantees.
- The root directory contains maintained code which should be a good starting
  point for experiments.

# Usage

- I work on throwaway experiments directly in `exp`.
- I occasionally put starter code in the root directory if (1) it's likely to be
  useful across many projects and (2) I'm willing to commit to maintaining it.
- For large projects / papers, I fork the repo.
- I try not to have 'dead code' in lib/ which isn't used by at least one
  root-directory experiment. This lets the root-dir experiments function as a
  de-facto test suite.