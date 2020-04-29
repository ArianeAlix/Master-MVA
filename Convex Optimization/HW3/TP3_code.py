import matplotlib.pyplot as plt
import numpy as np

def loss(Q,p,A,b,t,v):
    return t*(v.T @ Q @ v + p.T @ v)-np.sum(np.log(b-A@v),axis=0)

def centering_step(Q,p,A,b,t,v0,eps):
    precision=1000
    n_iter=0
    v=v0.astype('float')
    v_seq=[v]
 
    # Parameters of the backtracking
    alpha = 0.05
    beta = 0.95
    
    # We implement Newton's method with backtracking line search
    while precision>eps:
        ### Computing the grad and hessian
        grad_tf0=  t *( (2*Q @ v) + p)
        hessian_tf0 = 2* t * Q
        
        # For the barrier part
        grad_phi= np.sum(A/(b-A@v),axis=0).reshape(v.shape[0],1)
        hessian_phi=np.zeros((np.shape(Q)))
        for i in range(A.shape[0]):
            #reshaping to have good dimensions
            line=A[i,:].reshape(np.shape(A)[1],1)
            hessian_phi+=(line @ line.T)/((b-A@v)[i]**2)
        
        # Grad and Hessian of the total objective function
        grad = grad_tf0+grad_phi
        hessian = hessian_tf0 + hessian_phi
        
        # We store the current value of v to compare it later
        v_prec=v
        
        
        ### Backtracking : 
        # we reduce the step until v+step*direction respects the condition
        
        #initial step
        step=1
        # direction of the descent
        direction=(-np.linalg.pinv(hessian) @ grad)

        # we also check that it would not go out of the domain
        while (loss(Q,p,A,b,t,v+step*direction) > loss(Q,p,A,b,t,v) + alpha*step*grad.T @ direction) or (any(b-A@(v+step*direction) <=0)):
            step=beta*step
        
        
        ### We update v and we store it in the sequence
        v=v+step*direction
        v_seq.append(v)
        

        ### We stop if the values are close
        precision = np.linalg.norm(v-v_prec)
        
    return v_seq
    
    
    
def barr_method(Q,p,A,b,v0,eps,mu):
    m=A.shape[0] #nb of constraints
    t=1

    v=v0.astype('float')
    v_seq=[v0]
    
    # to plot the evolution of the gap
    f_v=[(v.T @ Q @ v + p.T @ v)[0,0]]
    
    # to plot the evolution of the precision criterion
    prec=[m/t]
    
    while (m/t >eps):
        # We update v with the final value of the centering step
        v=centering_step(Q,p,A,b,t,v,eps)[-1]
        
        # We store the values of v after each centering step
        v_seq.append(v)
        
        # we store the corresponding loss
        f_v.append((v.T @ Q @ v + p.T @ v)[0,0])
        
        # we update t
        t=t*mu
        prec.append(m/t)

    return v_seq,prec,abs(f_v-f_v[-1])#gap
    
    
    

###### Test

# Generating random values
X=np.random.randint(-10,10,(100,2))
y=np.random.randint(-10,10,(100,1))
lambd=10

# Building the corresponding matrices and vectors for the Quadratic Problem
Q=(1/2)*np.linalg.inv(X.T @ X)
p=np.linalg.inv(X.T @ X) @ X.T @ y
A=np.concatenate((-np.eye(2),np.eye(2)))
b=lambd*np.ones((4,1))

v0=np.array([[0.1,0.1]]).T
eps=10**(-8)


for mu in [2,15,50,100]:
    v_seq,prec,gap=barr_method(Q,p,A,b,v0,eps,mu)
    plt.figure(1)
    plt.title('Precision criterion')
    plt.xlabel('Nb of outer iterations')
    plt.ylabel('m/t')
    plt.plot(prec,label='mu='+str(mu))
    plt.legend()
    plt.show()

    gap=gap.reshape(gap.shape[0])
    plt.figure(2)
    plt.title('Loss gap')
    plt.xlabel('Nb of outer iterations')
    plt.ylabel('|f(vt)- f*|')
    plt.plot(gap,label='mu='+str(mu))
    plt.legend()
    plt.show()
    
    plt.figure(3)
    plt.title('Evolution of v (outer iterations results)')
    plt.xlabel('v_0')
    plt.ylabel('v_1')
    plt.plot(*zip(*v_seq),label='mu='+str(mu))
    plt.legend()
    plt.show()

    
    
    