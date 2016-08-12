import six.moves.cPickle as pickle
import gzip, os, sys, timeit, numpy
import theano, theano.tensor as T

# numpy.random.seed(1234)

n_in = 9 * 1024
n_out = 10
batch_size = 100

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W',borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def split_data(x, y, rtn=0.64, rvd=0.18, rtt=0.18):
    tnn, vdn, ttn = [int(len(y) * rr) for rr in [rtn, rvd, rtt]]
    train_set, valid_set, test_set = [x[0:tnn], y[0:tnn]], [x[tnn:tnn+vdn], y[tnn:tnn+vdn]], [x[tnn+vdn:tnn+vdn+ttn], y[tnn+vdn:tnn+vdn+ttn]]
    return train_set, valid_set, test_set


def load_data_():
    # load datasets
    x = numpy.array([]).astype('uint8')
    y = numpy.array([]).astype('uint8')
    nx = 0
    for i in range(1, 7):
        path = 'p2/data_batch_' + str(i)
        d = pickle.load(open(path, 'rb'))
        tx = d['data'].reshape(d['data'].shape[0] / 9, 9 * 1024)
        ty = numpy.array(d['labels']).astype('uint8')
        nx += tx.shape[0]
        x = numpy.append(x, tx)
        y = numpy.append(y, ty)
    x = x.reshape((nx, x.shape[0] / nx))
    
    # shuffle
    xy = numpy.array(zip(x, y))
    numpy.random.shuffle(xy)
    x, y = xy.transpose()

    # rearrange as uint 8 from object type
    nx = numpy.zeros((x.shape[0], n_in)).astype('uint8')
    for i, xx in enumerate(x):
        nx[i:i+1] = xx
    x = nx
    y = y.astype('uint8')
    
    train_set, valid_set, test_set = split_data(x, y)

    return train_set, valid_set, test_set

def data_to_tensor(train_set, valid_set, test_set):
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def load_data():
    train_set, valid_set, test_set = load_data_()
    rval = data_to_tensor(train_set, valid_set, test_set)
    return rval

def sgd_optimization_mnist(learning_rate=0.0013, n_epochs=100000, batch_size=batch_size):
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    classifier = LogisticRegression(input=x, n_in=n_in, n_out=n_out)

    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    print('... training the model')
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    improvement_threshold = 0.995  # a relative improvement of this much is
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f%%, rl %f' %
                    (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100., learning_rate))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best model %f%%') %
                        (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

                    # with open('best_model.pkl', 'wb') as f: pickle.dump(classifier, f)
                    classifier.W.get_value().tofile('sgd_w')
                    classifier.b.get_value().tofile('sgd_b')

                if this_validation_loss <= 0:
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best model %f%%') %
                        (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
                
            # if patience <= iter:
                # done_looping = True
                # break

    end_time = timeit.default_timer()
    print 'best validation', best_validation_loss * 100., ', test_score', test_score * 100.
    print 'epoch', epoch, ', ', 1. * epoch / (end_time - start_time), 'epochs/sec'
    print 'total time', (end_time - start_time), 'sec'
    classifier.W.get_value().tofile('sgd_e_w')
    classifier.b.get_value().tofile('sgd_e_b')

if __name__ == '__main__':
    sgd_optimization_mnist()

def test_samples(): # directly calculate from W, b
    def pred(x, w, b):
        o = numpy.dot(x, w) + b
        pred = numpy.argmax(o)
        return pred
        
    def ch_all(tset, w, b):
        res = [pred(x, w, b) for x in tset[0]]
        errs = list(res == tset[1]).count(False)
        print errs, '/', len(tset[0]), round(float(errs) / len(tset[0]), 4) * 100, '%'
        
    w = numpy.fromfile('sgd_w', dtype='float32').reshape((n_in, n_out))
    b = numpy.fromfile('sgd_b', dtype='float32')
    trs, vds, tts = load_data_()

    for st in [trs, vds, tts]: ch_all(st, w, b)

