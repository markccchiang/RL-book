'''Function approximations as affine spaces.

The code here follows book/appendix6 (Appendix F of the book), which is the
one appendix that is *about* the library: it argues that the parameters of a
function approximation form an affine space P, that the gradients form the
associated vector space G, and that a function approximation is a point that
a gradient translates.  The appendix points at the overloaded ``__add__`` of
``rl.function_approx.Gradient`` as where this shows up in code.

So rather than re-implement the algebra, this checks it against the classes
the appendix is describing.  Every law is verified on the real
``LinearFunctionApprox``, ``DNNApprox`` and ``Tabular``, whose parameter
containers D[R] are respectively a vector, a sequence of matrices and a
mapping — the genericity in D that the appendix insists on.

Two things worth knowing before reading the output:

  * ``Gradient.__add__`` is the appendix's two operations at once.  Gradient
    + Gradient is addition in the vector space G; Gradient + FunctionApprox
    is the affine ⊕, translating a point.  The translation has to be on the
    left, since Gradient is the class carrying the overload.
  * The appendix defines SGD(x, y)(p) = p ⊕ (α (y - f(x, p)) G(x)(p)), but
    ``FunctionApprox.update`` steps with Adam, not plain SGD.  Both are
    translations in G, so the affine structure is untouched; the specific
    translation is not the same one, as the last section shows.

Run it with:

    python -m rl.appendix6.affine_spaces
'''

from typing import Callable, Sequence, Tuple, TypeVar

import numpy as np

from rl.function_approx import (DNNApprox, DNNSpec, FunctionApprox,
                                Gradient, LinearFunctionApprox, Tabular,
                                Weights)

X = TypeVar('X')

# --- reading the parameters D[R] back out, whatever container D is ---------


def flat_params(approx: FunctionApprox) -> np.ndarray:
    '''The parameters of a function approximation, flattened to a vector.

    A point of P and a translation of G are both elements of D[R], so this
    works on either, and lets the laws below be checked with one function.
    '''
    if isinstance(approx, LinearFunctionApprox):
        return np.array(approx.weights.weights, dtype=float)
    if isinstance(approx, DNNApprox):
        return np.concatenate([w.weights.flatten() for w in approx.weights])
    if isinstance(approx, Tabular):
        return np.array([approx.values_map[k]
                         for k in sorted(approx.values_map)], dtype=float)
    raise TypeError(f'no parameter reader for {type(approx).__name__}')


def gap(left, right) -> float:
    '''max |difference| of the parameters of two points (or two
    translations).'''
    unwrap = (lambda g: g.function_approx) if isinstance(left, Gradient) \
        else (lambda a: a)
    return float(np.max(np.abs(flat_params(unwrap(left))
                               - flat_params(unwrap(right)))))


def least_squares_gradient(
        approx: FunctionApprox,
        xy_values: Sequence[Tuple[X, float]]
) -> Gradient:
    '''G(x)(p) for the usual objective, i.e. what update() would step along.'''
    return approx.objective_gradient(
        xy_values,
        lambda xs, ys: approx.evaluate(xs) - np.array(ys)
    )


def check(label: str, difference: float, tolerance: float = 1e-12) -> None:
    '''Print a law, the size of its violation, and whether it holds.'''
    verdict = 'holds' if difference <= tolerance else 'FAILS'
    print(f'  {label:54s}{difference:14.2e}   {verdict}')


if __name__ == '__main__':

    rng = np.random.default_rng(1729)
    features: Sequence[Callable[[float], float]] = [
        lambda x: 1., lambda x: x, lambda x: x ** 2, lambda x: np.sin(x)
    ]
    xs = [0.3, 0.8, 1.4, 2.2, 3.1]
    data = [(x, float(np.cos(x))) for x in xs]

    linear = LinearFunctionApprox.create(
        feature_functions=features,
        weights=Weights.create(weights=rng.normal(0., 1., 4))
    )
    deep = DNNApprox.create(
        feature_functions=features,
        dnn_spec=DNNSpec(
            neurons=[3], bias=True,
            hidden_activation=lambda a: np.maximum(a, 0.),
            hidden_activation_deriv=lambda h: np.where(h > 0., 1., 0.),
            output_activation=lambda a: a,
            output_activation_deriv=lambda o: np.ones_like(o)
        )
    )
    tabular: Tabular[float] = Tabular(values_map={x: float(np.cos(x))
                                                  for x in xs})

    print('\n1. The Gradient Space G is a vector space')
    print(f'  {"law":54s}{"violation":>14s}')
    for name, approx in [('LinearFunctionApprox  (D = vector)', linear),
                         ('DNNApprox  (D = sequence of matrices)', deep),
                         ('Tabular  (D = mapping)', tabular)]:
        u = least_squares_gradient(approx, data[:3])
        v = least_squares_gradient(approx, data[2:])
        w = least_squares_gradient(approx, data[1:4])
        print(f'  {name}')
        check('    u + v = v + u', gap(u + v, v + u))
        check('    (u + v) + w = u + (v + w)', gap((u + v) + w, u + (v + w)))
        check('    v + 0 = v', gap(v + v.zero(), v))
        check('    1 * v = v', gap(v * 1., v))
        check('    a * (b * v) = (a b) * v', gap((v * 3.) * 0.5, v * 1.5))
        check('    a * (u + v) = a * u + a * v',
              gap((u + v) * 2., u * 2. + v * 2.))
        check('    (a + b) * v = a * v + b * v',
              gap(v * (2. + 3.), v * 2. + v * 3.))

    print('\n2. The two meanings of Gradient.__add__')
    grad = least_squares_gradient(linear, data)
    other = least_squares_gradient(linear, data[:2])
    print(f'  Gradient + Gradient        -> '
          f'{type(grad + other).__name__:22s} a translation in G')
    print(f'  Gradient + FunctionApprox  -> '
          f'{type(grad + linear).__name__:22s} a point in R')
    try:
        # mypy rejects this statically too, hence the ignore: the operator
        # genuinely does not exist in this direction
        linear + grad          # type: ignore[operator]
    except AttributeError as problem:
        print(f'  FunctionApprox + Gradient  -> AttributeError: {problem}')
    print('  ⊕ takes the translation on the left, because Gradient carries '
          'the overload;')
    print('  mypy rejects the other direction statically, before it can '
          'raise')

    print('\n3. P and R are affine spaces')
    print(f'  {"law":54s}{"violation":>14s}')
    for name, approx in [('LinearFunctionApprox', linear),
                         ('DNNApprox', deep),
                         ('Tabular', tabular)]:
        v = least_squares_gradient(approx, data[:3])
        w = least_squares_gradient(approx, data[2:])
        print(f'  {name}')
        check('    a ⊕ 0 = a  (right identity)',
              gap(v.zero() + approx, approx))
        check('    (a ⊕ v) ⊕ w = a ⊕ (v + w)  (associativity)',
              gap(w + (v + approx), (v + w) + approx))
        # bijection: the unique translation carrying one point to another
        moved = (v * 2.5) + approx
        difference = Gradient(moved * 1. + approx * -1.)      # moved ⊖ approx
        check('    a ⊕ (b ⊖ a) = b  (⊖ inverts ⊕)',
              gap(difference + approx, moved))

    print('\n4. Where the affine structure lives: parameters, not values')
    probe = [0.5, 1.7, 2.9]
    print(f'  {"|I(p ⊕ v)(z) - (I(p)(z) + I(v)(z))| at z =":54s}'
          + ''.join(f'{z:10.2f}' for z in probe))
    for name, approx in [('LinearFunctionApprox', linear),
                         ('DNNApprox', deep)]:
        v = least_squares_gradient(approx, data)
        translated = (v + approx).evaluate(probe)
        separate = approx.evaluate(probe) + v.function_approx.evaluate(probe)
        print(f'  {name:54s}'
              + ''.join(f'{d:10.5f}' for d in np.abs(translated - separate)))
    print('  ⊕ is addition of parameters; only for a linear approximation '
          'does that')
    print('  also add the values, since evaluation is then linear in p')

    print('\n5. SGD is a translation, and for a linear approximation you can '
          'write it out')
    α = 0.1
    x, y = 1.4, float(np.cos(1.4))
    feature_x = linear.get_feature_values([x])[0]
    error = y - linear(x)
    step = Gradient(LinearFunctionApprox.create(
        feature_functions=features,
        weights=Weights.create(weights=α * error * feature_x)
    ))
    updated = step + linear
    print(f'  α = {α}, (x, y) = ({x}, {y:.6f}), '
          f'prediction error e(p) = {error:.6f}')
    print(f'  {"z":>10s}{"g(z)":>14s}{"g^(x,y)(z)":>16s}'
          f'{"difference":>14s}{"α e Φ(z)·Φ(x)":>18s}')
    for z in probe:
        feature_z = linear.get_feature_values([z])[0]
        predicted = α * error * float(np.dot(feature_z, feature_x))
        print(f'  {z:10.2f}{linear(z):14.6f}{updated(z):16.6f}'
              f'{updated(z) - linear(z):14.6f}{predicted:18.6f}')

    adam = linear.update([(x, y)])
    adam_step = flat_params(adam) - flat_params(linear)
    print('\n  the library steps with Adam, not with plain SGD:')
    print(f'  {"":22s}{"parameter step":>44s}')
    print(f'  {"plain SGD  α e Φ(x)":22s}'
          + ''.join(f'{c:11.6f}' for c in α * error * feature_x))
    print(f'  {"update() (Adam)":22s}'
          + ''.join(f'{c:11.6f}' for c in adam_step))
    print('  both are translations in G, so every law above still applies; '
          'they are just')
    print('  different translations')
    print()
