
import numpy as np
import matplotlib.pyplot as plt


X = np.arange(-10,10,0.05)
Y = 5*np.sin(X)*np.cos(0.5*X)
tau = 0.5



def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r--')

for tau in [5,.5,.05,.005]:
    plt.scatter(X, Y, c='b', s=2)
    for X_h in np.random.uniform(-10,10,25):
        # Method 1 (https://stats.stackexchange.com/a/348128/178089)
        w = np.exp(-(X_h-X)**2/(2*tau**2))
        a = np.sum(w)
        b = np.sum(w*X)
        c = np.sum(w*X**2)
        d = np.sum(w*Y)
        e = np.sum(w*X*Y)
        mat1 = np.array([[a,b],[b,c]])
        mat2 = np.array([d,e])
        theta1 = (mat1**-1).dot(mat2)

        # Method 2 (Direct Soln)
        X_m = np.stack([np.ones(len(X),), X]).T
        X_hm = [1, X_h]
        w = np.exp(-np.linalg.norm(X_m-X_hm,axis=1)**2/(2*tau**2)) # both w are same, but this is the general form
        X_dash = X_m * np.stack([np.sqrt(w), np.sqrt(w)]).T
        Y_dash = np.sqrt(w)*Y
        theta2 = ((X_dash.T.dot(X_dash))**-1).dot(X_dash.T).dot(Y_dash)

        # Verify
        print("Both solutions are same?", np.allclose(theta1, theta2))
        Y_pred = np.dot([1, X_h], theta2)
        plt.plot([X_h,],[Y_pred,],'x',c='r')
        # intercept, slope, = theta2
        # abline(slope, intercept)

    plt.savefig('plot_tau_{}.jpg'.format(tau))
    plt.close()


for tau in [5,.5,.05,.005]:
    X_h = np.random.uniform(-10,10,1)
    w = np.exp(-(X_h-X)**2/(2*tau**2))
    plt.plot(X, w, c='b')
    plt.savefig('local_weight_tau_{}.jpg'.format(tau))
    plt.close()
