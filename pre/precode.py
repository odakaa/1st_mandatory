# Einar Holsb\o done this atrocity 2017
import numpy as np

# Set to True to load the big data set instead
BIGDATA = False

print "Loading data..."

if BIGDATA:
    data = np.genfromtxt('creditfraud.csv', delimiter=',')
    lastcol = 29
    truth = [0.144800089, 4.537449930, 5.246915267, 7.115031732, 2.518657741,
             1.545085627, -2.830038407, -2.410473992, 34.786359072, 
             -9.919927002, 23.812173327, -61.670817783, 40.189039662, 
             33.862622820, -16.615272064, -0.618733376, 15.309273546, 
             5.631484914, -4.066823502, -0.888067511, -2.755034830, 
             -3.322881536, 0.162538846, -0.916925947, -0.288467553, 
             -0.456699696, -0.304502125,  0.008763563, 0.010576039]
else:
    data = np.genfromtxt('spam.csv', delimiter=',')
    lastcol = 57
    truth = [   -1.492773e-01,-2.405872e-01,-1.598794e-01, 1.792132e-02,
                1.521252e+00, 3.116990e-02, 5.117602e-01, 1.328038e-01, 
                9.483219e-02, 4.187917e-01, 7.968122e-01,-7.066990e-01, 
                -5.672777e-01,-4.229411e-02, 1.217488e+01,-6.331285e-02, 
                1.828883e+00, 2.489344e-02, 1.059606e-01, 3.472549e+00, 
                2.804220e-01, 5.623414e-01, 1.025395e+00, 5.151430e-01, 
                -8.775583e+00,-1.929012e-01,-1.384527e+03, 1.620060e+00, 
                4.331880e-01,-5.719369e-01,-9.152286e+01,-1.425805e+05, 
                -2.991609e-01, 1.430954e+05,-8.514605e+00, 2.089551e-01, 
                -1.557250e+00, 9.149123e+01, 4.663478e-01,-3.890767e-01, 
                -1.601655e+02,-7.735470e+01,-1.787178e+00,-2.385616e+01, 
                -6.600457e-01,-9.924051e-01,-1.279255e+00,-1.400962e+00, 
                -5.165253e-01,-1.250286e-02,-4.783125e-01, 1.081903e+00, 
                1.688540e+00,-1.991582e+00, 1.464189e+01,-3.194082e+00, 
                4.274010e+00]

# DEAR STUDENT: you might want to experiment with number of batches
batches = 100

print "Done."

# keep first 500 samples for validation
test = data[0:499, :]
train = data[500:, :]


# labels reside in the last column.
x = np.delete(train, lastcol, 1)
y = train[:, lastcol]

def prob_given_x(x, coefficients):
    return 1/(1 + np.exp(-np.dot(x,coefficients)))

# calculates the gradient wrt j
def gradient(x, y, coefficients, j):
    h = prob_given_x(x, coefficients)
    xj = x[:, j]
    grad = np.dot(np.subtract(h, y), xj)/len(x)
    return grad

# initial coefficient vector
coefs = [0.0]*lastcol

# learning rate
learning_rate = 0.005

batch_size = len(x)/batches

for iteration in range(500):
    # DEAR STUDENT: basically parallellize this loop
    for j in range(batches):
        subset_x = x[j*batch_size:(j+1)*batch_size, :]
        subset_y = y[j*batch_size:(j+1)*batch_size]

        for i in range(lastcol):
            coefs[i] = coefs[i] - learning_rate*gradient(subset_x, subset_y, coefs, i)

    diff = np.mean(abs(np.subtract(truth, coefs)))
    print" Iteration "+ str(iteration + 1) + " of 500. Diff=" + str(round(diff,5))


test_x = np.delete(test, lastcol, 1)
test_y = test[:, lastcol]

predictions = prob_given_x(test_x, coefs)
classes = [1 if x > .5 else 0 for x in predictions]

print "\n  accuracy=" + str(sum(classes == test_y)/float(len(classes)))

