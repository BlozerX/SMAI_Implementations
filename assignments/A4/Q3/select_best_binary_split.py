import numpy as np

def select_best_binary_split(x_NF, y_N, MIN_SAMPLES_LEAF=1):
    ''' Determine best single feature binary split for provided dataset
    COMPLETED VERSION
    '''
    N, F = x_NF.shape

    # Allocate space to store the cost and threshold of each feat's best split
    cost_F = np.inf * np.ones(F)
    thresh_val_F = np.zeros(F)

    for f in range(F):
        x_N = x_NF[:, f]
        
        # Compute all possible x threshold values for current feature
        xunique_U = np.unique(x_N)
        if xunique_U.size <= 1:
            continue # Can't split on a feature with only one unique value

        possib_xthresh_V = 0.5 * (xunique_U[:-1] + xunique_U[1:])
        V = possib_xthresh_V.size
        
        if V == 0:
            continue

        # Compute total cost at each possible threshold
        total_cost_V = np.inf * np.ones(V)
        left_yhat_V = np.zeros(V)
        right_yhat_V = np.zeros(V)

        for v_idx, thresh in enumerate(possib_xthresh_V):
            left_mask_N = x_N < thresh
            right_mask_N = ~left_mask_N
            
            y_L = y_N[left_mask_N]
            y_R = y_N[right_mask_N]

            # Enforce minimum leaf size
            if y_L.size < MIN_SAMPLES_LEAF or y_R.size < MIN_SAMPLES_LEAF:
                continue # Cost remains inf

            mean_L = np.mean(y_L)
            mean_R = np.mean(y_R)
            
            left_yhat_V[v_idx] = mean_L
            right_yhat_V[v_idx] = mean_R

            cost_L = np.sum(np.square(y_L - mean_L))
            cost_R = np.sum(np.square(y_R - mean_R))
            
            total_cost_V[v_idx] = cost_L + cost_R

        # Check if any valid split was found
        if not np.all(np.isinf(total_cost_V)):
            # Pick out the split candidate that has best cost
            chosen_v_id = np.argmin(total_cost_V)
            cost_F[f] = total_cost_V[chosen_v_id]
            thresh_val_F[f] = possib_xthresh_V[chosen_v_id]

    # Determine single best feature to use
    best_feat_id = np.argmin(cost_F)
    best_thresh_val = thresh_val_F[best_feat_id]
    
    if not np.isfinite(cost_F[best_feat_id]):
        # Edge case: not possible to split further
        return (None, None, None, None, None, None)

    ## Assemble the left and right child datasets
    left_mask_N = x_NF[:, best_feat_id] < best_thresh_val
    right_mask_N = np.logical_not(left_mask_N)
    x_LF, y_L = x_NF[left_mask_N], y_N[left_mask_N]
    x_RF, y_R = x_NF[right_mask_N], y_N[right_mask_N]

    return (best_feat_id, best_thresh_val, x_LF, y_L, x_RF, y_R)
