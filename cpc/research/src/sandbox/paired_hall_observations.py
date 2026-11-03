"""Two Hall observables with an explicit, supplied noise metric.

Log carrier density and Hall mobility reuse the same Hall coefficient. The
primitive coordinates below are log10(|RH|/[cm^3/C]) and log10(rho/[Ohm*cm]).
No covariance is estimated or certified from published tables by this module.
"""
import numpy as np
from .ionized_transport import E_CHARGE

PAIR_TO_PRIMITIVES=np.array([[-1.,0.],[-1.,-1.]])
PRIMITIVES_TO_PAIR=np.array([[-1.,0.],[1.,-1.]])


def hall_pair_primitives(log_nH,log_muH):
    pair=np.asarray([log_nH,log_muH],float)
    if not np.isfinite(pair).all():raise ValueError('Two finite log10 Hall observables required')
    return -np.log10(E_CHARGE)+PAIR_TO_PRIMITIVES@pair


def paired_hall_residual(predicted_pair,observed_pair,pair_jacobian,*,primitive_covariance):
    """Whiten paired residuals in a user-specified primitive covariance.

    Covariance entries use dex^2. A chosen metric is not a measured covariance
    or a calibrated likelihood unless its experimental evidence supports that.
    """
    pred=np.asarray(predicted_pair,float);obs=np.asarray(observed_pair,float)
    jac=np.asarray(pair_jacobian,float);cov=np.asarray(primitive_covariance,float)
    if pred.shape!=(2,) or obs.shape!=(2,) or jac.ndim!=2 or jac.shape[0]!=2 or cov.shape!=(2,2):
        raise ValueError('Explicit two-observable values,matching Jacobian and2x2 covariance required')
    if not all(np.isfinite(x).all() for x in (pred,obs,jac,cov)) or not np.allclose(cov,cov.T,rtol=0,atol=1e-14):
        raise ValueError('Finite symmetric covariance and finite observations required')
    factor=np.linalg.cholesky(cov)
    residual=np.linalg.solve(factor,PAIR_TO_PRIMITIVES@(pred-obs))
    derivative=np.linalg.solve(factor,PAIR_TO_PRIMITIVES@jac)
    return residual,derivative
