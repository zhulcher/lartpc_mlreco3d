import scipy
import numpy as np
import numba as nb

from .globals import LAR_X0, MUON_MASS,PION_MASS,KAON_MASS,PROT_MASS
from .energy_loss import step_energy_loss_lar

from mlreco.utils.tracking import get_track_segments


def mcs_fit(theta, M, dx, z = 1, \
        split_angle = False, res_a = 0.25, res_b = 1.25):
    '''
    Finds the kinetic energy which best fits a set of scattering angles
    measured between successive segments along a particle track.

    Parameters
    ----------
    theta : np.ndarray
        (N) Vector of scattering angle at each step in radians
    M : float
        Particle mass in MeV/c^2
    dx : float
        Step length in cm
    z : int, default 1
        Impinging partile charge in multiples of electron charge
    split_angle : bool, default False
        Whether or not to project the 3D angle onto two 2D planes
    res_a : float, default 0.25 rad*cm^res_b
        Parameter a in the a/dx^b which models the angular uncertainty
    res_b : float, default 1.25
        Parameter b in the a/dx^b which models the angular uncertainty
    '''
    # TODO: give a lower bound using CSDA (upper bound?)
    fit_min = scipy.optimize.minimize_scalar(mcs_nll_lar,
            args=(theta, M, dx, z, split_angle, res_a, res_b),
            bounds=[10., 100000.])

    return fit_min.x


@nb.njit(cache=True)
def mcs_nll_lar(T0, theta, M, dx, z = 1,
        split_angle = False, res_a = 0.25, res_b = 1.25):
    '''
    Computes the MCS negative log likelihood for a given list of segment angles
    and an initial momentum. This function checks the agreement between the
    scattering expection and the observed scattering at each step.

    Parameters
    ----------
    T0 : float
        Candidate particle kinetic energy in MeV
    theta : np.ndarray
        (N) Vector of scattering angle at each step in radians
    M : float
        Particle mass in MeV/c^2
    dx : float
        Step length in cm
    z : int, default 1
       Impinging partile charge in multiples of electron charge
    split_angle : bool, default False
        Whether or not to project the 3D angle onto two 2D planes
    res_a : float, default 0.25 rad*cm^res_b
        Parameter a in the a/dx^b which models the angular uncertainty
    res_b : float, default 1.25
        Parameter b in the a/dx^b which models the angular uncertainty
    '''
    # Compute the kinetic energy at each step
    assert len(theta), 'Must provide angles to esimate the MCS loss'
    num_steps = len(theta + 1)
    ke_array  = step_energy_loss_lar(T0, M, dx, num_steps=num_steps)

    # If there are less steps than expected, T0 is too low
    if len(ke_array) < num_steps + 1:
        return np.inf

    # Convert the kinetic energy array to momenta
    mom_array = np.sqrt(ke_array**2 + 2 * M * ke_array)

    # Define each segment momentum as the geometric mean of its end points
    mom_steps = np.sqrt(mom_array[1:] * mom_array[:-1])

    # Get the expected scattering angle for each step
    theta0 = highland(mom_steps, M, dx, z)

    # Update the expected scattering angle to account for angular resolution
    res = res_a/dx**res_b
    theta0 = np.sqrt(theta0**2 + res**2)

    # Compute the negative log likelihood, return
    if not split_angle:
        nll = np.sum(0.5 * (theta/theta0)**2 + 2*np.log(theta0))
    else:
        theta1, theta2 = split_angles(theta)
        nll = np.sum(0.5 * (theta1/theta0)**2 \
                + 0.5 * (theta2/theta0)**2 + 2*np.log(theta0))

    return nll


@nb.njit(cache=True)
def highland(p, M, dx, z = 1, X0 = LAR_X0):
    '''
    Highland scattering formula

    Parameters
    ----------
    p : float
       Momentum in MeV/c
    M : float
       Impinging particle mass in MeV/c^2
    dx : float
        Step length in cm
    z : int, default 1
       Impinging partile charge in multiples of electron charge
    X0 : float, default LAR_X0
       Radiation length in the material of interest in cm

    Results
    -------
    float
        Expected scattering angle in radians
    '''
    # Highland formula
    beta = p / np.sqrt(p**2 + M**2)
    prefactor = 13.6 * z / beta / p

    return prefactor * np.sqrt(dx/LAR_X0) * \
            (1. + 0.038*np.log(z**2 * dx / LAR_X0 / beta**2))


@nb.njit(cache=True)
def split_angles(theta):
    '''
    Split 3D angles between vectors to two 2D projected angles

    Parameters
    ----------
    theta : np.ndarray
        Array of 3D angles in [0, np.pi]

    Returns
    -------
    float
        Frist array of 2D angle in [-np.pi, np.pi]
    float
        Second array of 2D angle in [-np.pi, np.pi]
    '''
    # Randomize the azimuthal angle
    phi = np.random.uniform(0, 2*np.pi, len(theta))
    offset = (theta > np.pi/2) * np.pi
    sign = (-1)**(theta > np.pi/2)

    # Project the 3D angle along the two planes
    radius = np.sin(theta)
    theta1 = offset + sign * np.arcsin(np.sin(phi) * radius)
    theta2 = offset + sign * np.arcsin(np.cos(phi) * radius)

    # If theta is over pi/2, must flip the angles
    return theta1, theta2



def get_track_angles(points,start,dx=5):
    if len(points)==0: return []
    _, dirs, _ = get_track_segments(points, dx, start)
    costh = np.sum(dirs[:-1] * dirs[1:], axis = 1)
    costh = np.clip(costh, -1, 1)
    return np.arccos(costh)
        
def MCS_Mass(theta,KE,dx=5): 
    #for some reason the scipy minimizer is not working well :( 
    #This is a super hacky way around it 

    #almost certainly need to correct for missing energy in calo KE vs truth KE

    err=np.inf
    retmass=np.nan
    for mass in np.logspace(1, 3.1, num=50):
        new_err=mcs_nll_lar(KE, theta, mass, dx)
        if new_err<err: retmass=mass
    return retmass
def best_MCS_Mass(theta,KE,dx,test=[MUON_MASS,PION_MASS,PROT_MASS,KAON_MASS]):
    err=np.inf
    retmass=np.nan
    for mass in test:
        new_err=mcs_nll_lar(KE, theta, mass, dx)
        if new_err<err: retmass=mass
    return retmass

def proj_dEdx(points,depositions,start,end):
    if np.linalg.norm(end-start)==0: return None,None
    
    start=np.array(start)
    end=np.array(end)
    projlist=[]
    for i in points:
        pt=np.array(i)
        projlist+=[np.dot(end-pt,end-start)/np.linalg.norm(end-start)] #distance from the end point
    projlist=np.array(projlist)
    E=np.array(depositions.copy())
    projlist=projlist[E>0]
    E=E[E>0]
    
    E=np.array([x for _, x in sorted(zip(projlist, E))])
    projlist=np.array(sorted(projlist))
    
    return projlist,E

def PIDA_integrated(projlist,E):
    return np.median(projlist**(.42-1)*(1-.42)*E)
