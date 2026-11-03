"""Analytic counterexamples and nuisance-projected information for the user proposal.

These are mathematical checks, not new Ga2O3 material predictions.
"""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'proposal_followup_20261021_v1/theory_checks'


def projected_design(J,weights=None,scales=None):
    J=np.asarray(J,float);k,p=J.shape
    weights=np.ones(k) if weights is None else np.asarray(weights,float)
    scales=np.ones(p) if scales is None else np.asarray(scales,float)
    whiten=np.sqrt(weights);a=whiten
    P=np.eye(k)-np.outer(a,a)/(a@a)
    K=P@(whiten[:,None]*J*scales[None,:])
    return K,K.T@K,P


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol_before.json').exists()
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        purpose='counterexamples to unconditional monotonicity/precision guarantees;working-information algebra',
        code_hashes=c.file_hashes([Path(__file__)]),
        settings=dict(designs=[[[1.,2.,3.]],[[1.,2.,3.],[2.,5.,8.]],[[1.,2.],[2.,3.],[4.,6.]]],
            coordinate_rescale=[1e3,1e-3],prior_scales=[.05,.4],two_state_transport_weights=[.5,.5]),
        no_material_data_or_new_fits=True,actual_noise_covariance_known=False))
    records=[]
    def check(name,condition,**values):
        records.append(dict(name=name,passed=bool(condition),**values))
    for J in [np.array([[1.,2.,3.]]),np.array([[1.,2.,3.],[2.,5.,8.]]),np.array([[1.,2.],[2.,3.],[4.,6.]])]:
        K,F,P=projected_design(J);rank=int(np.linalg.matrix_rank(K,tol=1e-12))
        check('offset_profiled_rank_bound_'+str(len(J)),rank<=len(J)-1,
            anchors=len(J),raw_rank=int(np.linalg.matrix_rank(J)),profiled_rank=rank,
            effective_eigenvalues=np.linalg.eigvalsh(F).tolist(),raw_max_eigenvalue=float(np.linalg.eigvalsh(J.T@J)[-1]))
        check('offset_projection_'+str(len(J)),np.max(abs(P@np.ones(len(J))))<1e-12 and np.max(abs(P@P-P))<1e-12)
    J=np.array([[1.,3.],[2.,1.],[4.,2.]])
    w=np.array([1.,2.,3.]);s=np.array([.05,.4]);D=np.array([1e3,1e-3])
    _,F1,_=projected_design(J,w,s);_,F2,_=projected_design(J*D,w,s/D)
    _,raw1,_=projected_design(J,w);_,raw2,_=projected_design(J*D,w)
    check('prior_whitened_information_coordinate_invariance',np.max(abs(F1-F2))<1e-12,
        scaled_max_error=float(np.max(abs(F1-F2))),raw_eigenvalues_before=np.linalg.eigvalsh(raw1).tolist(),
        raw_eigenvalues_after=np.linalg.eigvalsh(raw2).tolist())
    k=2;J=np.array([[1.,3.],[2.,4.]]);y=np.array([.3,-.2]);alpha=.7
    K,F,P=projected_design(J)
    z=np.linalg.solve(F+alpha*np.eye(2),K.T@(P@y));delta=float(np.mean(y-J@z))
    M=np.block([[np.array([[float(k)]]),np.ones((1,k))@J],
        [J.T@np.ones((k,1)),J.T@J+alpha*np.eye(2)]])
    joint=np.linalg.solve(M,np.r_[np.sum(y),J.T@y])
    check('profiled_and_joint_linear_solution',np.max(abs(joint-np.r_[delta,z]))<1e-12,
        max_error=float(np.max(abs(joint-np.r_[delta,z]))))
    # Counterexamples establish that the proposed universal claims cannot hold.
    true_query=0.;base=0.;noisy_anchor=1.;shift=noisy_anchor-base
    check('counterexample_no_realized_mae_monotonicity',abs(base-true_query)<abs(base+shift-true_query),
        zero_anchor_error=0.,one_anchor_error=1.,exact_shared_model=True)
    check('counterexample_shift_does_not_inherit_unit_slope',abs(2.*1.-1.)>0.,
        truth='y=x',shared_curve='f=2x',anchor_x=0.,query_x=1.,shared_plus_shift_error=1.,unit_dose_error=0.)
    tau0=np.array([1.,1.]);tau1=np.array([1.,3.]);weights=np.array([.5,.5])
    rh0=float(weights@(tau0*tau0)/(weights@tau0)**2);rh1=float(weights@(tau1*tau1)/(weights@tau1)**2)
    check('counterexample_fixed_free_n_not_fixed_Hall_n',abs(rh0-rh1)>.1,
        free_n=1.,rH_before=rh0,rH_after=rh1,Hall_n_before=1/rh0,Hall_n_after=1/rh1)
    # r=(theta1-a,theta1+theta2-b). Masked gradient=(r1,r2)
    # has mixed partials 0 and 1, unlike the actual least-squares gradient.
    masked_field_jac=np.array([[1.,0.],[1.,1.]])
    check('counterexample_masked_jacobian_is_not_joint_loss_derivative',not np.allclose(masked_field_jac,masked_field_jac.T),
        mixed_partial_g1_theta2=0.,mixed_partial_g2_theta1=1.,
        valid_alternative='conditional block optimization or actual constrained step with full derivatives')
    F=np.diag([100.,.1]);b=np.ones(2);gate=100./101.
    update=gate*np.linalg.solve(F+np.eye(2),b)
    check('counterexample_scalar_gate_not_principal_eigenvector_projection',abs(update[1])>0.,update=update.tolist())
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'checks.json',dict(created_utc=c.e.now(),passed=all(r['passed'] for r in records),
        checks=len(records),records=records,actual_material_fits=0,scope='analytic/linear algebra counterexamples only',
        stable_prediction_gain_established=False))
    print(dict(passed=all(r['passed'] for r in records),checks=len(records)))


if __name__=='__main__':main()
