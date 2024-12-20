
def dynmodes(Nsq, depth, nmodes):
    """
    DYNMODES calculates ocean dynamic vertical modes
    taking a column vector of Brunt-Vaisala values (Nsq) at
    different pressures (p) or depth and calculating some number of 
    dynamic modes (nmodes). 

    Note: The input pressures need not be uniformly spaced, 
           and the deepest pressure is assumed to be the bottom.

    USAGE:   [wmodes,pmodes,ce]=dynmodes(Nsq,depth,nmodes);

    Inputs:  Nsq    = column vector of Brunt-Vaisala buoyancy frequency (s^-2)
    		 depth  = column vector of water depth (m, negative values)
             nmodes = number of vertical modes to calculate 
               
    Outputs: wmodes = vertical velocity structure
             pmodes = horizontal velocity structure
             ce     = modal speed (m/s)

    developed by huanglongyu (hly24@foxmail.com) 2024.12.20
    referred to John Klinck. July 1999
    """
    from scipy import linalg
    import numpy as np

    p = -depth
    rho0 = 1028

    # Check for surface value
    n = len(p)
    if p[0] > 0:
        # Add surface pressure with top Nsq value
        z = np.zeros(n+1)
        z[1:] = -p
        N2 = np.zeros(n + 1)
        N2[0] = Nsq[0]
        N2[1:] = Nsq
        nz = n + 1
    else:
        z = -p
        N2 = Nsq
        nz = n

    # Calculate depths and spacing
    dz = z[:-1] - z[1:]
    # Midpoint depth
    zm = z[:-1] - 0.5 * dz
    # Midpoint spacing
    dzm = np.zeros(nz)
    dzm[1:nz-1] = zm[:nz-2] - zm[1:nz-1]
    dzm[0] = dzm[1]
    dzm[nz-1] = dzm[nz-2]

    # Get dynamic modes
    A = np.zeros((nz, nz))
    B = np.zeros((nz, nz))
    # create matrices
    for i in range(1, nz-1):
        A[i,i] = 1/(dz[i-1]*dzm[i]) + 1/(dz[i]*dzm[i])
        A[i,i-1] = -1/(dz[i-1]*dzm[i])
        A[i,i+1] = -1/(dz[i]*dzm[i])
    for i in range(nz):
        B[i,i] = N2[i]
    # Set boundary conditions
    A[0,0] = -1
    A[nz-1,0] = -1

    # Solve eigenvalue problem
    eigvals, wmodes = linalg.eig(A,B)
    eigvals = np.real(eigvals)

    # Process eigenvalues and eigenvectors
    ind = np.array(np.where(np.imag(eigvals) == 0))[0]
    eigvals = eigvals[ind]
    wmodes = wmodes[:, ind]

    ind = np.array(np.where(np.real(eigvals) >= 1e-10))[0]
    eigvals = eigvals[ind]
    wmodes = wmodes[:, ind]

    ind = np.argsort(eigvals)
    eigvals = eigvals[ind]
    wmodes = wmodes[:, ind]

    # Normalize modes
    for i in range(wmodes.shape[1]):
        norm = wmodes[:,i] / np.max(np.abs(wmodes[:,i]))
        wmodes[:,i] = norm

    # Phase speed
    nm = len(eigvals)
    ce = 1 / np.sqrt(eigvals)

    # create pressure structure
    pmodes = np.zeros(wmodes.shape)
    for i in range(nm):
        # calculate first deriv of vertical modes
        pr = np.diff(wmodes[:,i])
        pr[:nz-1] = pr[:nz-1] / dz[:nz-1]
        pr = pr * rho0 *ce[i] * ce[i]
        # linearly interpolate back to original depths
        pmodes[1:nz-1,i] = 0.5 * (pr[1:nz-1] + pr[:nz-2])
        pmodes[0,i] = pr[0]
        pmodes[nz-1,i] = pr[nz-2]
    
    # delete -2 line to keep the same shape with depth
    wmodes = np.delete(wmodes, -2, axis=0)
    pmodes = np.delete(pmodes, -2, axis=0)

    return wmodes[:,:nmodes], pmodes[:,:nmodes], ce[:nmodes]
