"""Full Sandbox-response learning through analytic implicit derivatives.

An energy-trained RBF model is refined with a regularized nonlinear property
loss. Damped Gauss-Newton proposals are accepted only after evaluating the full
loss. The finite-iteration algorithm does not certify a global optimum.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.metrics.pairwise import rbf_kernel
from .energy_learning import group_weights
from .property_metric_learning import PropertyMetricKernel


class NonlinearPropertyKernel(PropertyMetricKernel):
    def __init__(self,length=1.,alpha=.01,strength=1.,max_steps=6,max_lines=4):
        super().__init__(length=length,alpha=alpha,metric="energy",strength=0.)
        if strength<0 or max_steps<0 or max_lines<1:
            raise ValueError("Invalid refinement budget")
        self.property_strength=float(strength)
        self.max_steps=int(max_steps);self.max_lines=int(max_lines)

    def prepare(self,records):
        super().fit(records)
        self.phi=rbf_kernel(self.X,self.X,gamma=self.gamma)@self.projector
        self.centered_targets=np.array([r["crossing_eV"] for r in records])-self.mean
        self.energy_weights=group_weights(records)
        self.energy_weights/=self.energy_weights.sum()
        self.energy_gram=(self.phi.T*self.energy_weights)@self.phi+self.alpha*np.eye(len(self.beta))
        self.energy_rhs=self.phi.T@(self.energy_weights*self.centered_targets)
        self.energy_beta=self.beta.copy()
        return self

    def loss_and_gradient(self,beta,response,derivatives=True):
        """response(xi,derivatives) -> scaled residual, dresidual/dxi, metadata."""
        beta=np.asarray(beta,float)
        xi=self.mean+self.phi@beta
        residual,J,meta=response(xi,derivatives)
        residual=np.asarray(residual,float)
        if not np.isfinite(residual).all():
            raise ValueError("Nonfinite nonlinear response residual")
        error=self.phi@beta-self.centered_targets
        energy_loss=float(self.energy_weights@(error**2)+self.alpha*(beta@beta))
        property_loss=float(self.property_strength*(residual@residual))
        loss=energy_loss+property_loss
        if not derivatives:
            return loss,None,None,dict(energy_loss=energy_loss,property_loss=property_loss,**meta)
        J=np.asarray(J,float)
        if J.shape!=(len(residual),len(xi)) or not np.isfinite(J).all():
            raise ValueError("Invalid implicit response Jacobian")
        A=J@self.phi
        gradient=2*(self.energy_gram@beta-self.energy_rhs+self.property_strength*A.T@residual)
        GN=2*(self.energy_gram+self.property_strength*A.T@A)
        return loss,gradient,GN,dict(energy_loss=energy_loss,property_loss=property_loss,**meta)

    def fit_nonlinear(self,records,response):
        self.prepare(records)
        beta=self.beta.copy();trace=[];calls=0;accepted=0;stop="max_steps"
        for iteration in range(self.max_steps):
            calls+=1
            try:
                loss,gradient,H,meta=self.loss_and_gradient(beta,response,True)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                stop="initial_or_iterate_response_failure"
                trace.append(dict(iteration=iteration,error=repr(exc),accepted=False))
                break
            gnorm=float(np.linalg.norm(gradient,np.inf))
            row=dict(iteration=iteration,objective=loss,gradient_inf=gnorm,
                energy_loss=meta["energy_loss"],property_loss=meta["property_loss"],
                response_metadata=meta,trials=[])
            if gnorm<1e-5:
                stop="small_gradient";row["accepted"]=False;trace.append(row);break
            damping=max(1e-12,1e-8*float(np.trace(H))/len(H))
            direction=-cho_solve(cho_factor(H+damping*np.eye(len(H)),lower=True),gradient)
            slope=float(gradient@direction)
            if not slope<0:
                stop="no_descent_direction";row["accepted"]=False;trace.append(row);break
            accepted_trial=None
            for j in range(self.max_lines):
                step=2.**(-j)
                calls+=1
                try:
                    trial,_,_,tmeta=self.loss_and_gradient(beta+step*direction,response,False)
                    armijo=float(loss+1e-4*step*slope)
                    good=bool(trial<=armijo+1e-12)
                    row["trials"].append(dict(step=step,objective=trial,
                        armijo_upper=armijo,accepted=good))
                    if good:
                        accepted_trial=(step,trial);break
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    row["trials"].append(dict(step=step,error=repr(exc),accepted=False))
            row["damping"]=damping;row["accepted"]=accepted_trial is not None
            trace.append(row)
            if accepted_trial is None:
                stop="line_search_budget_exhausted";break
            step,trial=accepted_trial
            beta+=step*direction;accepted+=1
            if (loss-trial)<=1e-7*max(1.,abs(loss)):
                stop="small_relative_decrease";break
        self.beta=beta;self.dual=self.projector@self.beta
        self.diagnostics.update(nonlinear_refinement=True,
            property_strength=self.property_strength,max_steps=self.max_steps,max_lines=self.max_lines,
            accepted_steps=accepted,response_batch_calls=calls,stop_reason=stop,
            optimization_trace=trace,
            claim="accepted full-loss descent only; not global optimality or material validation")
        return self
