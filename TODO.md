## V4:
    - In case, if a token is getting extended from a node, and if the node later has a most confident timestep, but if somehow the extended token's path becomes the most probable, then the timestep for the node with prev token will be pointing to later part, which should be before the extended token, but it won't...

## V5:
    - Included the creation of new node in case of a more confident occurance of repeat, but getting segmentation fault.