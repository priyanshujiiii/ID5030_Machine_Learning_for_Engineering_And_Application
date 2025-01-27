import numpy as np
import pandas as pd


try:
    X=int(input())
    Y=int(input())
    Z=int(input())
    def Inter_roll_no(x,y,z):
        # x=int(input())
        # y=int(input())
        # z=int(input())
        X=[[1,10,100],[1,25,625],[1,40,1600],[1,60,3600],[1,80,6400],[1,85,7225]]
        Y=[[1/(9+0.1*x)],[1/(6+0.01*y)],[1/(4.3+0.01*z)],[1/3.01],[1/2.22],[1/2.07]]
        XT = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
        XTX = np.dot(XT,X)
        XTXinv = np.linalg.inv(XTX)
        XTY = np.dot(XT,Y)
        w = np.dot(XTXinv,XTY)
        a=(1/w[2])
        b=(a*w[1])
        c=(a*w[0])
        return (a,b,c)

    def q2(X,Y,Z):
        # X=int(input())
        # Y=int(input())
        # Z=int(input())
        pr = [9+0.1*X , 6+0.01*Y, 4.3+0.01*Z, 3.01, 2.22, 2.07]
        y = [1/pr[i] for i in range(6)]
        T = [10, 25, 40, 60, 80, 85]
        Tmax = max(T)
        Tmin = min(T)
        x = [(T[i] - Tmin)/(Tmax - Tmin) for i in range(6)] 
        w0 = 0.1
        w1 = 0.1
        w2 = 0.1
        alpha = 0.1
        dJdw0 = (1/6)*(-sum(y) + w0*6 + w1*sum(x) + w2*sum([x[i]**2 for i in range(6)]))
        dJdw1 = (1/6)*(-sum([x[i]*y[i] for i in range(6)]) + w0*sum(x) + w1*sum([x[i]**2 for i in range(6)]) + w2*sum(x[i]**3 for i in range(6)))
        dJdw2 = (1/6)*(-sum([x[i]**2*y[i] for i in range(6)]) + w0*sum([x[i]**2 for i in range(6)]) + w1*sum([x[i]**3 for i in range(6)]) + w2*sum(x[i]**4 for i in range(6)))
        dJ = [dJdw0, dJdw1, dJdw2]
        W = [w0 - alpha*dJ[0], w1 - alpha*dJ[1], w2 - alpha*dJ[2]]
        return (W[0],W[1],W[2])
    
    def q3(X,Y,Z):
        # X=int(input())
        # Y=int(input())
        # Z=int(input())
        pr = [9+0.1*X , 6+0.01*Y, 4.3+0.01*Z, 3.01, 2.22, 2.07]
        y = [1/pr[i] for i in range(6)]
        T = [10, 25, 40, 60, 80, 85]
        Tmax = max(T)
        Tmin = min(T)
        x = [(T[i] - Tmin)/(Tmax - Tmin) for i in range(6)] 
        w = [0.1, 0.1, 0.1] # initialize weights
        alpha = 0.01 # learning rate
        epochs = 1 # number of epochs
        for epoch in range(epochs):
            for i in range(len(x)):
                print(i,w[0],w[1],w[2])
                # compute gradients for each data point
                dJdw0 = -y[i] + w[0] + w[1]*x[i] + w[2]*x[i]**2
                dJdw1 = (-y[i]*x[i]) + w[0]*x[i] + w[1]*x[i]**2 + w[2]*x[i]**3
                dJdw2 = (-y[i]*x[i]**2) + w[0]*x[i]**2 + w[1]*x[i]**3 + w[2]*x[i]**4
                # update weights
                w[0] = w[0] - alpha*dJdw0
                w[1] = w[1] - alpha*dJdw1
                w[2] = w[2] - alpha*dJdw2
                
        return (w[0], w[1], w[2])
    q_1 = Inter_roll_no(X,Y,Z)
    q_2 = q2(X,Y,Z)
    q_3=q3(X,Y,Z)
    print(q_1)
    print()
    
    print(q_2)
    print()

    print(q_3)
        
except:
    raise ValueError('please enter a valid roll_no')