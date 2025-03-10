## V6:
    - Write tests to ensure the CTC logic.
    - If at t=N, there is a diff in parent prob, we include that into 'squash_prob'
        also at t=(N+1), if the same parent has changed or unchanged probs, we still
        are including that diff prob to 'squash_prob'. ISSUE
