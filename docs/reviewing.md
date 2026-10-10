# Reviewing this library: the job is to break it

You are not asked to confirm that it works. Every check so far was written by the author or by AI models of the same kind as the author, so whatever they all miss is still in here.
A review that finds nothing is useful only if it says exactly what was tried.

Start with `docs/verification_status.md`: it lists what was checked, how, and what is known not to be (section "Not verified"). Attack the claims there.

## Four jobs, in order of value per hour. Pick the ones that fit your time.

**A. Refute three claims (about 1 hour).** Read the table "How things were checked" and the list "Known simplifications, with their size". Choose three statements you doubt and write an input
that makes each one false: a panel, a signal, a parameter combination. A wrong size ("drift cost 2 to 3 percent") counts as much as a wrong sign.

**B. Look for look-ahead and survivorship (about 2 hours).** The engine says that a signal at the close of day d earns the return from close(d + lag) to close(d + lag + 1).
Try to find a path by which information from after d reaches a position held at d: eligibility, delisting flags, funding, the neutralising controls, the benchmark alignment, resampling in the
review functions. `tests/test_causality.py` and `tests/test_metamorphic.py` show the kind of test that would have caught one.

**C. Reproduce one number somewhere else (about 2 hours).** Take one result (a Sharpe ratio, a drawdown, a cost drag, a Brinson term, a Newey-West t) on a small panel and reproduce it by hand,
in a spreadsheet or with another library. Report the first digit where it differs, even if you think the other tool is wrong.

**D. Find tests that do not test (about 1 hour).** `python docs/automut.py pitbacktest/robustness.py --n 40` breaks the code in small ways, one at a time (about 10 minutes per 25 mutants with 6 cores), and lists the ones no test noticed. Look at the survivors:
a harmless change (an equivalent mutant) is fine, a change that alters a result and still passes every test is a finding. Or do it by hand: change one comparison, sign or constant in a function of
your choice and run `python tests/test_<file>.py`.

## What to send back

One entry per finding, as small as possible:

1. What you believe is wrong, in one sentence.
2. The shortest code that shows it (a seeded random panel is fine), and what it prints.
3. What you expected instead and why (a formula, a reference, another tool).
4. How sure you are, and what would change your mind.

Also send: what you tried that held up, with the time it took. "I tried X and could not break it" is a result.

Things the author already knows and lists as limits (order book, margin, Korean dividends, Windows) are not findings unless the documents under-state them.
