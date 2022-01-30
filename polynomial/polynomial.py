"""Module for manipulating n-dimensional polynomials."""
from functools import reduce
from operator import mul, add
from itertools import product

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.special import perm


class Polynomial:
    """Power-series representation of an n-dimensional polynomial.
    Supports array evaluation, addition, multiplication, composition,
    derivation, and integration."""

    __slots__ = ["_coef"]

    def __init__(self, coef):
        self._coef = np.asanyarray(coef)

    @classmethod
    def interpolate(cls, points, values):
        """Returns the interpolating polynomial through the provided points."""
        points = np.asanyarray(points)
        values = np.asanyarray(values)
        if points.shape[0] != values.shape[0]:
            raise ValueError("Points and values have different lengths")
        if len(values.shape) != 1:
            raise ValueError("Values should be a one-dimensional array")
        if len(points.shape) != 2:
            raise ValueError("Points should be a two-dimensional array")
        dimension = points.shape[1]
        return reduce(
            add,
            (
                values[i] * reduce(
                    mul,
                    (
                        reduce(
                            mul,
                            (
                                cls(
                                    np.array([-point2[dim], 1]).reshape(
                                        tuple(
                                            2 if k == dim else 1
                                            for k in range(dimension))
                                    )
                                ) / (point1[dim] - point2[dim])
                                for j, point2 in enumerate(points)
                                if not np.allclose(point1[dim], point2[dim])
                            ),
                            cls.one(dimension)
                        )
                        for dim in range(dimension)
                    ),
                    cls.one(dimension)
                )
                for i, point1 in enumerate(points)
            ),
            cls.zero(dimension)
        )

    @classmethod
    def lagrange(cls, points):
        """Returns lagrange interpolating polynomials at points"""
        l = len(points)
        return [
            cls.interpolate(points, [int(i==j) for j in range(l)])
            for i in range(l)
        ]

    def __call__(self, arg):
        arg = np.asanyarray(arg)
        if arg.shape[-1] != self.dimension:
            raise ValueError(
                "Degree mismatch: cannot evaluate a polynomial of dimension "
                f"{self.dimension} on arguments of first dimension "
                f"{arg.shape[0]}"
            )
        if arg.dtype == np.dtype('O'):
            return self._poly_call(arg)
        return self._vector_call(np.moveaxis(arg, -1, 0))

    def __repr__(self):
        coef = repr(self.coef)[6:-1]
        name = self.__class__.__name__
        return f"{name}({coef})"

    def __add__(self, other):
        """Adds two polynomials. Resulting polynomial is as large as either"""
        if not isinstance(other, Polynomial):
            raise ValueError("Cannot add non-polynomial to polynomial")
        if other.dimension != self.dimension:
            raise ValueError(
                "Cannot add polynomials with incompatible dimensions"
            )
        new_shape = [max(l, r) for l, r in zip(self.shape, other.shape)]
        new_dtype = np.find_common_type((self.dtype, other.dtype), ())
        new_coef = np.zeros(new_shape, new_dtype)
        slc1 = tuple(slice(None, d) for d in self.shape)
        new_coef[slc1] = self.coef
        slc2 = tuple(slice(None, d) for d in other.shape)
        new_coef[slc2] += other.coef
        return self.__class__(new_coef)

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            if other.dimension != self.dimension:
                raise ValueError(
                    "Polynomial dimension mismatch in multiplication"
                )
            return self._prod(other)
        return self._scale(other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._scale(1/other)

    def deriv(self, arg):
        """Derivative with respect to arg.

        Examples
        ========
        >>> from polymomial import Polynomial
        >>> p = Polynomial([[1, 2, 3], [4, 5, 6]])
        >>> p.deriv([0, 1])
        Polynomial([[2, 6], [5, 12]])
        """
        arg = np.asanyarray(arg)
        if len(arg.shape) != 1:
            raise NotImplementedError(
                "Must derive with dim 1 arrays"
            )
        if arg.shape[0] != self.dimension:
            raise ValueError(
                "Incompatible derivative dimension"
            )
        slc = tuple(slice(d, None) for d in arg)
        indices = Polynomial._index_array(self.shape)
        factors = perm(indices, arg).prod(axis=-1)[slc].astype(np.int32)
        return Polynomial(factors * self.coef[slc])

    def integ(self, arg):
        """Integrate by args.

        Examples
        ========
        >>> from polymomial import Polynomial
        >>> p = Polynomial([[1, 2, 3], [4, 5, 6]])
        >>> p.integ([0, 1])
        Polynomial([[0, 1, 1, 1], [0, 4, 2.5, 2]])
        """
        arg = np.asanyarray(arg)
        if len(arg.shape) != 1:
            raise NotImplementedError(
                "Must integrate with dim 1 arrays"
            )
        if arg.shape[0] != self.dimension:
            raise ValueError(
                "Incompatible integral dimension"
            )
        indices = Polynomial._index_array(self.shape)
        factors = 1 / perm(indices + arg, arg).prod(axis=-1)
        new_shape = np.asarray(self.shape) + arg
        result = np.zeros(new_shape)
        for index in indices.reshape((-1, indices.shape[-1])):
            print(index)
            result[tuple(index + arg)] = (
                factors[tuple(index)] * self.coef[tuple(index)]
            )
        return Polynomial(result)

    @classmethod
    def zero(cls, dimension):
        """
        Construct a polynomial of desired dimension with all zero coefficients.
        """
        return cls(np.zeros((1,)*dimension))

    @classmethod
    def one(cls, dimension):
        """
        Construct a polynomial of desired dimension equal to the constant 1.
        """
        return cls(np.ones((1,)*dimension))

    @property
    def coef(self):
        """Coefficients of the polynomial as an ndarray.
        Example: [[1, 2], [3, 4]] represents the polynomial
        1 + 2x + 3 y + 4 xy"""
        return self._coef

    @property
    def shape(self):
        "Shape of the polynomial"
        return self.coef.shape

    @property
    def dtype(self):
        "Data type of the polynomial coefficients"
        return self.coef.dtype

    @property
    def dimension(self):
        "Number of variables of the polynomial"
        return len(self.coef.shape)

    @property
    def degree(self):
        """The maximum single power. For instance, 1 + x + y + xy
        would be degree 1, not 2"""
        return max(self.coef.shape)

    def _scale(self, fac):
        """Scales a polynomial by a number."""
        return self.__class__(self.coef * fac)

    def _prod(self, other):
        new_shape = [l + r - 1 for l, r in zip(self.shape, other.shape)]
        new_dtype = np.find_common_type((self.dtype, other.dtype), ())
        my_indices = (
            Polynomial._index_array(self.shape).reshape(-1, self.dimension)
        )
        other_indices = (
            Polynomial._index_array(other.shape).reshape(-1, other.dimension)
        )
        result = np.zeros(new_shape, new_dtype)
        for my_index, other_index in product(my_indices, other_indices):
            result[tuple(my_index + other_index)] += (
                self.coef[tuple(my_index)] * other.coef[tuple(other_index)]
            )
        return self.__class__(result)

    @staticmethod
    def _index_array(shape):
        """Construct an array of shape (*self.shape, len(self.shape))
        where the last axis represents the index of the element
        e.g. if shape = (3, 2) constructs
        [[[0, 0], [0, 1]], [[1, 0], [1, 1]], [[2, 0], [2, 1]]]"""
        return np.moveaxis(np.indices(shape), 0, -1)

    @staticmethod
    def _trim_zero(array):
        "Drop zero coefficients for highest powers"
        slices = tuple(slice(None, idx.max() + 1) for idx in np.nonzero(array))
        return array[slices]

    def _poly_call(self, arg):
        "Compose polynomial"
        poly_iter = np.nditer(
            self.coef, op_flags=['readonly'], flags=['multi_index']
        )
        # TODO get dimension robustly
        dim = arg[0].dimension
        if len(arg.shape) != 1:
            # TODO Support multidimensional polynomial arrays
            raise NotImplementedError(
                "Can't compose on multdimensional polynomial arrays"
            )
        result = reduce(
            add,
            (
                coef * reduce(
                    mul,
                    (
                        reduce(
                            mul,
                            (arg[i] for _ in range(power)),
                            Polynomial.one(dim)
                        )
                        for i, power in enumerate(poly_iter.multi_index)
                    ),
                    Polynomial.one(dim)
                )
                for coef in poly_iter
            ),
            Polynomial.zero(dim)
        )
        return Polynomial(Polynomial._trim_zero(result.coef))

    def _vector_call(self, arg):
        "Vectorized array call"
        it = iter(arg)
        x0 = next(it)
        c = polyval(x0, self.coef, tensor=True)
        for xi in it:
            c = polyval(xi, c, tensor=False)
        return c

