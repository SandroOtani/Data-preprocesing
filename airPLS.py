from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse import csr_matrix, eye, diags
from tqdm import tqdm
import numpy as np

def airPLS(df, lambda_, porder, itermax, wep, p):
    '''
    Baseline correction using adaptive iteratively reweighted Penalized Least Squares
    Code adapted from the MATLAB version of airPLS proposed by Zhang
    Input: 
    	df: Dataframe of spectra or chromatogram (size mxn, m is the number of samples and n is the number of variables)
    	lambda: lambda is an adjustable parameter, it can be adjusted by user. The larger lambda is, the smoother z will be 
    	order: an integer indicating the order of the difference of penalties
    	wep: weight exception proportion at both the start and end
    	p: asymmetry parameter for the start and end
    	itermax: maximum iteration times
    Output:
    	Xc: the corrected spectra or chromatogram vector (size m*n)
    	Z: the fitted vector (size m*n)
    Example:
    	Xc, Z = airPLS(df, lambda_ = 10e9, porder = 3, itermax = 20, wep = 0.1, p = 0.05);
    Sandro K. Otani 09/06/21
    Sílvia Claudino Martins Gomes
    Reference:
   	[1] Z.-M. Zhang. https://github.com/zmzhang/airPLS
    '''
    X=np.array(df)
    m=X.shape[0]
    n=X.shape[1]
    wi1=np.arange(0,np.ceil(wep*n))
    wi2=np.arange(np.floor(n-n*wep)-1,n)
    wi=list(np.concatenate((wi1,wi2), axis=0).astype('int'))
    E=np.eye(n,n)
    E=np.diff(E,n=porder)
    DD=lambda_*np.dot(E,E.T)
    Z=np.zeros((m,n))
    for i in tqdm(range(m)):
        w=np.ones(n)
        x=X[i,:]
        for j in range(1,itermax+1):
            W = diags(w ,offsets = 0,shape = (n,n))
            C=np.linalg.cholesky(W+DD).T   
            z=np.linalg.solve(C,np.linalg.solve(C.T,w*x))
            d=x-z
            dssn=np.abs(d[d<0].sum())
            if(dssn<(0.001*np.abs(x).sum())): 
                break
            w[d>=0]=0
            w[wi]=p
            w[d<0] = j*np.exp(abs(d[d<0])/dssn)
        Z[i,:]=z
    Xc = X-Z
    return Xc, Z 
