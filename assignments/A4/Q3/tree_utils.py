"""
tree_utils.py

Defines two Python classes, one for each kind of nodes:
- InternalDecisionNode
- LeafNode

COMPLETED VERSION
"""

import numpy as np

class InternalDecisionNode(object):
    '''
    Defines a single node used to make yes/no decisions within a binary tree.
    '''

    def __init__(self, x_NF, y_N, feat_id, thresh_val, left_child, right_child):
        self.x_NF = x_NF
        self.y_N = y_N
        self.feat_id = feat_id
        self.thresh_val = thresh_val
        self.left_child = left_child
        self.right_child = right_child


    def predict(self, x_TF):
        ''' Make prediction given provided feature array
        For an internal node, we assign each input example to either our
        left or right child to get its prediction.
        We then aggregate the results into one array to return.
        '''
        T, F = x_TF.shape

        # Create an empty array to store predictions
        yhat_T = np.zeros(T, dtype=np.float64)

        # Determine which of the input T examples belong to the
        # left child and which belong to the right
        left_mask_T = x_TF[:, self.feat_id] < self.thresh_val
        right_mask_T = ~left_mask_T

        # Ask the left child for its predictions (call 'predict')
        # We only pass the examples that belong to the left child
        if np.any(left_mask_T):
            yhat_T[left_mask_T] = self.left_child.predict(x_TF[left_mask_T])
        
        # Ask the right child for its predictions (call 'predict')
        if np.any(right_mask_T):
            yhat_T[right_mask_T] = self.right_child.predict(x_TF[right_mask_T])
        
        # Aggregate all predictions and return one array
        return yhat_T


    def __str__(self):
        ''' Pretty print a string representation of this node
        '''
        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()
        lines = [
            "Decision: X[%d] < %.3f?" % (self.feat_id, self.thresh_val),
            "  Y: " + left_str.replace("\n", "\n    "),
            "  N: " + right_str.replace("\n", "\n    "),
            ]
        return '\n'.join(lines)


class LeafNode(object):
    '''
    Defines a single node within a binary tree that makes constant predictions.
    '''

    def __init__(self, x_NF, y_N):
        self.x_NF = x_NF
        self.y_N = y_N


    def predict(self, x_TF):
        ''' Make prediction given provided feature array
        For a leaf node, all input examples get the same predicted value.
        '''
        T = x_TF.shape[0]
        
        # The prediction for a regression tree leaf is the mean of the y values
        if self.y_N.size == 0:
            y_pred = 0.0
        else:
            y_pred = np.mean(self.y_N)
            
        # Return one array with this prediction repeated T times
        yhat_T = np.full(T, y_pred)
        return yhat_T


    def __str__(self):
        ''' Pretty print a string representation of this node
        '''        
        if self.y_N.size == 0:
            return "Leaf: predict y = 0.000 (0 examples)"
        return "Leaf: predict y = %.3f (%d examples)" % (np.mean(self.y_N), self.y_N.size)
