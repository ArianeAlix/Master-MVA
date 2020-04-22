import numpy as np
import pandas as pd

### Computation of ranks from signatures or phenotypes tables
def ranks(table,preview=False):
    
    ### Computation of ranks
    N_gene=table.shape[0]

    R=pd.DataFrame(np.zeros(table.shape),columns=table.columns,index=table.index)

    if preview:
        p=0
    else:
        p=1
        
    for d in (table.columns):
        #Genes sorted by absolute value
        genes=table.loc[:,[d]]
        genes['abs']=np.abs(genes.loc[:,:])
        genes['sgn']=np.sign(genes.loc[:,[d]])

        genes=genes.sort_values('abs',ascending=True)

        #Computation of signed rank
        genes['rank']=np.arange(1,N_gene+1)*genes['sgn']
        genes=genes.drop(['abs','sgn'],axis=1)

        #One preview of ranks for a drug or disease
        if p==0:
            print('Preview of ranked genes for a drug:')
            display(genes.head(10))
            p+=1

        #Storing everything in the rank matrix
        R[d]=genes.loc[:,['rank']]
        

    return R




### Similarity metric as in the ssCMap method
def sscmap_sim(table):
    R=ranks(table,preview=False)
    
    
    #Initialization
    N_g=R.shape[0]
    N_d=R.shape[1]

    S=pd.DataFrame(np.zeros((N_d,N_d)),columns=R.columns,index=R.columns)
    
    
    # Computation of the similarity
    for d1 in (R.columns):
        for d2 in (R.columns):
            S[d1][d2] = int(R.loc[:,[d1]].values.T @ R.loc[:,[d2]].values)
            
            # Divided by max possible value
            max_val=np.sort(R.loc[:,[d1]].values,axis=0).T @ np.sort(R.loc[:,[d2]].values,axis=0)
            
            
            S[d1][d2] /= max_val
            
            #print(d1,d2,S[d1][d2])
        
    return S




def compute_sim_connections(A,S_drug,A_keep,S_keep):
    S_td=pd.DataFrame(np.zeros((A.shape[0],A.shape[0])),columns=A.index,index=A.index)
    
    t=0
    for i in (A.index[:A_keep]): # We iterate over the diseases in the index, that are not in rows filled with 0
        
        if t%10==0 or t==len(A.index[:A_keep])-1:
            print(t/len(A.index[:A_keep])*100,'%')        
        t+=1
        
        for j in (A.index[:A_keep]):
            coeff_norm=0
            
            for l in (S_drug.columns[:S_keep]):
                
                if A[l][i]!=0:
                
                    for k in (S_drug.columns[:S_keep]):
                        
                        if A[k][j]!=0:
                            #Numerator
                            S_td[j][i]+= A[l][i]*A[k][j]*S_drug[k][l]

                            # Coeff to normalize : sum of max possible connections : denominator
                            coeff_norm+= A[l][i]*A[k][j]
            S_td[j][i]/=coeff_norm
            
    return S_td


def weight(A,S,lambd):
    W=pd.DataFrame(np.zeros(S.shape),columns=S.columns,index=S.index)
    t=0
    for i in (A.index):
        if t%10==0 or t==len(A.index)-1:
            print(t/len(A.index)*100,'%')     
        t+=1
        
        order_i=np.sum(A.loc[i,:]) #order of disease i
        for j in (A.index):
            order_j=np.sum(A.loc[j,:]) #order of disease j
            for l in (A.columns):
                order_l=np.sum(A.loc[:,l]) #order of drug l
                
                W[j][i]+= A[l][i]*A[l][j]/order_l
            
            W[j][i]*= S[j][i]/( (order_i**(1-lambd)) * (order_j**lambd))
                
    return W