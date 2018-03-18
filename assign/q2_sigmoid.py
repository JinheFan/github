import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    
    ### YOUR CODE HERE

    dist = (1/(1 + np.exp(-x)))
    
    ### END YOUR CODE
    
    return dist

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x. 
    """
    
    ### YOUR CODE HERE
    dist = f*(1-f)
    ### END YOUR CODE
    
    return dist

def test_sigmoid_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(x)
    print f
    assert np.amax(f - np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-6
    print g
    assert np.amax(g - np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-6
    print "You should verify these results!\n"

def test_sigmoid(): 
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    x = np.array([[2, 5], [-2, -5]])
    f = sigmoid(x)
    g = sigmoid_grad(x)
    print f
    assert np.amax(f - np.array([[0.88079708, 0.99330715],
        [0.11920292, 0.00669285]])) <= 1e-6
    print g
    assert np.amax(g - np.array([[0.11920292, 0.04742587],
        [0.88079708, 0.99330715]])) <= 1e-6
    print "You should verify these results!\n"
    ### END YOUR CODE

if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
