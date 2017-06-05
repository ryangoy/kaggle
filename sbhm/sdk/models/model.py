class Model(object):
    """
    Abstract class for the basic structure for a stacked
    model that implements k-fold cross validation.
    """
    def __init__(self):
        raise NotImplementedError('Model subclass must impelement init.')

    def train(self, X_trn, y_trn):
        raise NotImplementedError('Model subclass must implement train.')

    def test(self, X_test, y_test=None):
        raise NotImplementedError('Model subclass must implement test.')


