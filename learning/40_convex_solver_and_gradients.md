# 40 - Convex Dual Solver and Implicit Differentiation

## Learning Objectives

- derive the fixed-covariance dual problem;
- prove uniqueness under positive covariance;
- understand Newton-CG matrix-vector products;
- implement and test the sensor adjoint;
- verify gradients before end-to-end training.

## 1. Primal problem

For fixed positive-definite \(\Sigma\), solve:

\[
\min_{0<x<1}
D_{\mathrm{Bern}}(x\|\sigma(v))
+
\frac12(y-Cx)^\top\Sigma^{-1}(y-Cx).
\]

Introduce residual \(\eta\):

\[
\min_{x,\eta}
D_{\mathrm{Bern}}(x\|\sigma(v))
+
\frac12\eta^\top\Sigma^{-1}\eta
\]

subject to:

\[
Cx+\eta=y.
\]

## 2. Lagrangian and stationarity

Choose:

\[
\mathcal{L}
=
D_{\mathrm{Bern}}(x\|\sigma(v))
+
\frac12\eta^\top\Sigma^{-1}\eta
-
\lambda^\top(Cx+\eta-y).
\]

Stationarity in \(x\):

\[
\operatorname{logit}(x)-v-C^\top\lambda=0,
\]

so:

\[
x(\lambda)=\sigma(v+C^\top\lambda).
\]

Stationarity in \(\eta\):

\[
\Sigma^{-1}\eta-\lambda=0,
\]

so:

\[
\eta(\lambda)=\Sigma\lambda.
\]

The constraint becomes:

\[
C\sigma(v+C^\top\lambda)+\Sigma\lambda-y=0.
\]

## 3. Dual objective

The corresponding convex dual minimization is:

\[
\phi(\lambda)
=
\sum_j
\operatorname{softplus}
\left(
v_j+(C^\top\lambda)_j
\right)
+
\frac12\lambda^\top\Sigma\lambda
-
y^\top\lambda.
\]

Gradient:

\[
\nabla\phi(\lambda)
=
Cx(\lambda)+\Sigma\lambda-y.
\]

Hessian:

\[
H(\lambda)
=
C
\operatorname{diag}
\left[
\sigma'(v+C^\top\lambda)
\right]
C^\top
+
\Sigma.
\]

## 4. Why the solution is unique

For nonzero vector \(a\):

\[
a^\top H a
=
\left\|
D^{1/2}C^\top a
\right\|_2^2
+
a^\top\Sigma a,
\]

where:

\[
D=\operatorname{diag}[\sigma'(\cdot)].
\]

The first term is nonnegative. If:

\[
\Sigma\succ0,
\]

then:

\[
a^\top\Sigma a>0.
\]

Therefore:

\[
H\succ0.
\]

The dual is strictly convex and has at most one minimizer. Unlike the hard
limit, this conclusion does not require \(C\) to have full row rank.

## 5. Why the dual is computationally useful

For one RGB patch:

- HR primal variables:
  \(3\times512\times512=786{,}432\);
- LR dual variables:
  \(3\times128\times128=49{,}152\).

The dual is approximately sixteen times smaller in scalar dimension.

It also requires only applications of:

- \(C^\top\);
- sigmoid;
- \(C\);
- diagonal covariance.

No dense matrix should be constructed.

## 6. Newton-CG update

At iteration \(k\), solve:

\[
H(\lambda_k)\Delta_k
=
-\nabla\phi(\lambda_k)
\]

using conjugate gradients, then update:

\[
\lambda_{k+1}
=
\lambda_k+\rho_k\Delta_k.
\]

Use damping or line search when necessary.

The Hessian-vector product for \(q\) is:

\[
Hq
=
C
\left[
D(C^\top q)
\right]
+
\Sigma q.
\]

This avoids explicit Hessian storage.

## 7. Stopping criteria

Track at least:

\[
r_{\mathrm{dual}}
=
\left\|
\nabla\phi(\lambda)
\right\|,
\]

\[
r_{\mathrm{decomp}}
=
\left\|
Cx+\Sigma\lambda-y
\right\|,
\]

and relative versions normalized by \(\lVert y\rVert+\epsilon\).

Stop when:

- relative decomposition residual is below tolerance;
- dual-gradient norm is below tolerance;
- maximum iterations is reached;
- numerical failure occurs.

Always return a convergence flag. Never silently treat a maximum-iteration
output as an exact solution.

## 8. Sensor forward and adjoint

The clean operator must expose:

```python
mu = operator.forward(hr)
hr_grad = operator.adjoint(lr_tensor)
```

Test:

\[
\epsilon_{\mathrm{adj}}
=
\frac{
\left|
\langle Cx,z\rangle-\langle x,C^\top z\rangle
\right|
}{
\lvert\langle Cx,z\rangle\rvert
+
\lvert\langle x,C^\top z\rangle\rvert
+
\epsilon
}.
\]

Run this test for:

- multiple blur strengths;
- multiple shapes;
- every channel count;
- CPU float64;
- GPU float32;
- boundary-heavy inputs.

## 9. Warm starts

Useful initializations:

- \(\lambda=0\) for the first sample;
- previous diffusion sample's \(\lambda\);
- previous patch with similar degradation;
- previous training iteration for deterministic validation.

Warm starts reduce iterations but must not leak target data or make baseline
runtime comparisons unfair.

## 10. Implicit differentiation

Let the optimum satisfy:

\[
F(\lambda^*,v,y,\Sigma)=0,
\]

where:

\[
F=C\sigma(v+C^\top\lambda^*)+\Sigma\lambda^*-y.
\]

For parameter \(q\):

\[
\frac{\partial\lambda^*}{\partial q}
=
-
\left(
\frac{\partial F}{\partial\lambda}
\right)^{-1}
\frac{\partial F}{\partial q}.
\]

The inverse is not formed explicitly. Backward uses another linear solve with
the Hessian.

## 11. Gradient validation

Before large training, compare:

1. finite differences;
2. autograd through fully unrolled iterations;
3. custom implicit backward.

Use tiny synthetic operators, such as:

- one-channel 4 by 4 HR;
- two-times block averaging;
- diagonal covariance;
- random logits away from saturation.

Report relative gradient error for:

- logits \(v\);
- observation \(y\);
- covariance diagonal;
- optional operator parameters.

Do not connect the solver to diffusion training until these tests agree.

## 12. Numerical failure modes

### Saturated logits

When \(x\) approaches 0 or 1:

\[
\sigma'(z)\to0.
\]

Positive \(\Sigma\) still stabilizes the Hessian, but sensitivity and finite
precision can worsen.

### Tiny covariance

Very small variance approaches the hard problem and can increase conditioning
cost. Use a positive floor and report it.

### Incorrect adjoint

CG may still return values with a wrong transpose, but they do not solve the
intended optimization problem. The adjoint test is mandatory.

### Clamp inside \(C\)

Clamp makes the operator nonlinear and invalidates the displayed dual
derivation.

## Exercises

1. Derive \(x(\lambda)\) and \(\eta(\lambda)\) from stationarity.
2. Prove \(H\succ0\) when \(\Sigma\succ0\).
3. Write the Hessian-vector product without constructing a matrix.
4. Explain why the noisy dual is smaller than the HR primal.
5. Design a finite-difference gradient test.

## Mastery Checklist

- [ ] I can derive the dual objective, gradient, and Hessian.
- [ ] I know why positive covariance guarantees uniqueness.
- [ ] I can explain Newton-CG and its operator calls.
- [ ] I understand the forward-adjoint identity.
- [ ] I will validate implicit gradients before full training.

Next: [41 - Codebase Implementation Blueprint](41_implementation_blueprint.md).
