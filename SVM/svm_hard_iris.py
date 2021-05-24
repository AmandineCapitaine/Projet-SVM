
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import cvxpy as cp

iris=datasets.load_iris()

#répartition des données observées 
red = []
redy = []
blue = []
bluy = []
for i in range(len(iris.data)):
    if (iris.target[i] == 0):
        red += [iris.data[i,0]]
        redy += [iris.data[i,1]]
    elif (iris.target[i] == 1):
        blue += [iris.data[i,0]]
        bluy += [iris.data[i,1]]
        
        
x = []
y = []
for i in range(len(iris.target)):
    if (iris.target[i] == 1) or (iris.target[i] == 0):
        x += [iris.data[i,0:2].tolist()]
        if iris.target[i] == 0:
            y += [-1]
        else:
            y += [1]

x = np.array(x)
y = np.array(y)

e = -np.ones(len(x))
a = np.zeros((len(x)))

#Lagrangien
n=len(a)
G = np.zeros((n,n))
for i in range(n):
    for j in range(len(x)):
        G[i,j]=y[i]*y[j]*np.dot(x[i],x[j])
            
def lagrangien(a, x, y):
    L= ((1/2)*np.transpose(a) @ G @ a) + (np.dot(e,a))
    return L

#Gradient
def lagranJ(a,x,y):
    J = np.zeros(len(a))
    for i in range(len(a)):
        J[i] -= 1.
        for j in range(len(a)):
            J[i] += a[j]*G[i,j]
    return J


a = np.zeros(len(x))
h = 1e-4
fig, ax = plt.subplots()
line = np.linspace(0,10,150)
plane = np.zeros(150)
planep = np.zeros(150)
planem = np.zeros(150)


#
# Fonctions pour une descente de gradient "manuelle" (Utilise cvxpy pour l'espace des contraintes) ##########################################
#

#itération en hard SVM
def grad_iter_h (a, x, y, F, J, h):
    b = a - h*J(a,x,y)
    
    c = cp.Variable(shape=(len(b)))
    cost = cp.sum_squares(c - b)
    objective = cp.Minimize(cost)
    constraints = [c >= 0, c.T @ y == 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    b=c.value
    return b

#descente complète
def grad_descent (a, x, y, F, J, h):
    i = 0
    import imageio
    name = 'mes_images'
    images = []
    while (np.linalg.norm(J(a,x,y)) > 1) and (i < 100):
        #recherche du pas optimal
        jac = J(a,x,y)
        m = F(a - h*jac, x, y)
        # for n in range(-50,250):
        #     r = F(a - jac*(10**(-n*0.1)), x, y)
        #     if (r < m):
        #         m = r
        #         h = 10**(-n*0.1)
        
        #iteration descente de gradient
        
        a = grad_iter_h(a,x,y,F,J,h)
        i += 1
        
        #affectation de w, b, et plane (la droite sur le plot)
        w =  np.zeros(len(x[0]))
        b = 0.
        for n in range(len(x)):
            if a[n] != 0.:
                w += a[n]*y[n]*x[n]
    
        b = 0
        for n in range(len(x)):
            r = (y[n]/abs(y[n]))*((1/y[n])-np.dot(x[n],w))
        if r > abs(b) :
            if y[n] < 0:
                b = -r
            else:
                b = r
        
        for n in range(150):
            plane[n] = -(w[0]*line[n] + b)/w[1]
            #<w,x> + b = 1
            # w1*x1 + w2*x2 + b = 1
            planep[n] = -(w[0]*line[n] + b)/w[1] + 1/w[1]
            planem[n] = -(w[0]*line[n] + b)/w[1] - 1/w[1]
        
        print(w, b, "gradient : ", np.linalg.norm(jac), " loss : ", m, "  n°", i)
                
        #clear puis affichage du nouveau plot
        fig.clear(True)
        plt.plot(red, redy, 'r+')
        plt.plot(blue, bluy, 'b+')
        plt.plot(line, plane, 'gold')
        plt.plot(line, planep, ':', color='gold')
        plt.plot(line, planem,  ':', color='gold')
        plt.savefig(name+str(i)+'.png')
        images.append(imageio.imread(name+str(i)+'.png'))
        
        #print(a)
    imageio.mimsave('demo.gif', images, fps = 30)
    #print(np.linalg.norm(J(a,x,y)))
    return a

print("a = ", a)
#a = grad_descent(a,x,y,lagrangien,lagranJ,h)
print("a = ", a)

##############################################################################################################################################


#
# Version 100% CVXPY #########################################################################################################################
#
def solveCVXPY(x,y):
    a = cp.Variable(shape=(len(x)))
    print(a.size)
    objective = cp.Minimize(-cp.sum(a) + (1/2)*cp.quad_form(a,G))

    constraints = [a >= 0, a.T @ y == 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(prob.value)
    print(result)
    print(a.value)
    a = a.value

    #affectation de w, b, et plane (la droite sur le plot)
    w =  np.zeros(len(x[0]))
    b = 0.
    for n in range(len(x)):
         w += a[n]*y[n]*x[n]

    
#     Calcul de b
#     for n in range(len(x)):
#         r = (y[n]/abs(y[n]))*((1/y[n])-np.dot(x[n],w))
#         if r > abs(b) :
#             if y[n] < 0:
#                 b = -r
#             else:
#                 b = r
    
    # Calcul de b (d'après Julien Kaszuba)
    Support = (a > 1e-4).flatten()
    b = y[Support] - np.dot(x[Support], w)
    b=b[0]

    for n in range(150):
        plane[n] = -(w[0]*line[n] + b)/w[1]
        #<w,x> + b = 1
        # w1*x1 + w2*x2 + b = 1
        planep[n] = -(w[0]*line[n] + b)/w[1] + 1/w[1]
        planem[n] = -(w[0]*line[n] + b)/w[1] - 1/w[1]



    #clear puis affichage du nouveau plot
    low = 45
    high = -25
    fig.clear(True)
    plt.plot(red, redy, 'r+')
    plt.plot(blue, bluy, 'b+')
    plt.plot(line[low:high], plane[low:high], 'gold')
    plt.plot(line[low:high], planep[low:high], ':', color='gold')
    plt.plot(line[low:high], planem[low:high],  ':', color='gold')
    

#solveCVXPY(x,y)
a = grad_descent(a,x,y,lagrangien,lagranJ,h)






