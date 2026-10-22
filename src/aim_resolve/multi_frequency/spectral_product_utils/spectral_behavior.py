# Copyright(C) 2025
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors:  Julian Rüstig

from abc import ABC, abstractmethod
from functools import reduce

import jax.numpy as jnp
from jax.typing import ArrayLike
from nifty.re.model import Model
from nifty.re.tree_math.vector import Vector

from .scaled_excitations import ScaledExcitations


class HarmonicLogSpectralBehavior(Model, ABC):
    @property
    @abstractmethod
    def relative_log_frequencies(self):
        """The spectral axis is fixed to be the first index of the output of
        the model. The `relative_log_frequencies` cast the output arrays of the
        models to conform with this convention by following the numpy casting
        conventions."""
        pass

    @abstractmethod
    def mean(self, p) -> list[ArrayLike]:
        pass

    @abstractmethod
    def fluctuations(self, p) -> list[ArrayLike]:
        pass

    @abstractmethod
    def fluctuations_with_frequencies(self, parameters) -> ArrayLike:
        pass

    @abstractmethod
    def mean_with_frequencies(self, parameters) -> ArrayLike:
        pass

    @abstractmethod
    def remove_degeneracy_of_spectral_deviations(
        self, deviations: ArrayLike
    ) -> ArrayLike:
        pass


class SingleHarmonicLogSpectralBehavior(HarmonicLogSpectralBehavior, ABC):
    """Abstract base class for a model with single spectral behavior.
    Hence, mean is a single number and fluctations is an Array.

    Note
    ----
    This can be used to speed up some calculations in the
    `CorrelatedMultiFrequencySky` class.
    """

    pass


class SpectralIndex(SingleHarmonicLogSpectralBehavior):
    def __init__(
        self,
        log_frequencies: ArrayLike,
        mean: Model,
        spectral_scaled_excitations: ScaledExcitations,
        reference_frequency_index: int,
    ):
        """Spectral Index spectral behavior. Special case of the
        `SpectralPolynomial`. However, since it has a single parameter, it can
        be used to speed up evaluations.

        .. math::
            \\log F(\\nu, A, ) = A^{spec.}
            * \\left\\{ \\mathrm{fluc.}^A \\xi^A \\nu  \\right\\}
            + \\mathrm{mean}^A \\nu

        Parameters
        ----------
        log_frequencies: ArrayLike
            The log of the frequencies. The relative frequencies are calculated
            internally. See `reference_frequency_index`.
        mean: Model
            The means of the polynomial model. See equation above.
        spectral_scaled_excitations: Model
            The fluctuations already applied to the xis. Hence, the
            combination of \\mathrm{fluc.}^A \\xi^A.
        reference_frequency_index: int
            The index of the reference frequency. Used in order to calculate
            the relative frequencies.
        """
        # NOTE : The model assumes that spectral axis is fixed to be the first
        # index of the output model. The slicing tuple matches the shape of the
        # relative_log_frequencies using the numpy slicing convention.
        slicing_tuple = (slice(None),) + (None,) * len(
            spectral_scaled_excitations.target.shape
        )

        self._relative_log_frequencies = (
            log_frequencies - log_frequencies[reference_frequency_index]
        )[slicing_tuple]

        self._mean = mean
        self.spectral_scaled_excitations = spectral_scaled_excitations

        self._denominator = 1 / jnp.sum(self.relative_log_frequencies**2)

        super().__init__(
            domain=self._mean.domain | self.spectral_scaled_excitations.domain
        )

    @property
    def relative_log_frequencies(self):
        return self._relative_log_frequencies

    def mean(self, p) -> ArrayLike:
        return self._mean(p)

    def fluctuations(self, p) -> ArrayLike:
        return self.spectral_scaled_excitations(p)

    def fluctuations_with_frequencies(self, p) -> ArrayLike:
        return self.fluctuations(p) * self.relative_log_frequencies

    def mean_with_frequencies(self, p) -> ArrayLike:
        return self.mean(p) * self.relative_log_frequencies

    def remove_degeneracy_of_spectral_deviations(
        self, deviations: ArrayLike
    ) -> ArrayLike:
        dev_slope = (
            jnp.sum(deviations * self.relative_log_frequencies, axis=0)
            * self._denominator
        )
        return deviations - dev_slope * self.relative_log_frequencies

    def __call__(self, p):
        # NOTE : only needed for instantiation of the model. However, maybe
        # there is a better way !!!
        # Since the apply is not needed
        return None


class SpectralIndexCurvature(HarmonicLogSpectralBehavior):
    def __init__(
        self,
        log_frequencies: ArrayLike,
        mean: Model,
        spectral_scaled_excitations: ScaledExcitations,
        curvature_mean: Model,
        curvature_scaled_excitations: ScaledExcitations,
        reference_frequency_index: int,
    ):
        """Spectral index and spectral curvature, anchored at the reference
        frequency. Extends `SpectralIndex` by a curvature term:

        .. math::
            \\log F(\\nu) = A^{spec.} * \\left\\{ \\mathrm{fluc.}^\\alpha \\xi^\\alpha u
            + \\mathrm{fluc.}^\\beta \\xi^\\beta q(u) \\right\\}
            + \\mathrm{mean}^\\alpha u + \\mathrm{mean}^\\beta q(u),

        with :math:`u = \\log(\\nu / \\nu_\\mathrm{ref})` and
        :math:`q(u) = u^2 - (\\sum u^3 / \\sum u^2)\\, u`, i.e. :math:`u^2`
        orthogonalised against :math:`u` over the model frequencies. Both basis
        functions vanish at the reference frequency, so the reference-frequency
        map is unchanged; since :math:`q \\perp u`, :math:`\\alpha` is the slope
        of the line through the reference point (as in `SpectralIndex`) and
        :math:`\\beta` is the coefficient of :math:`u^2` (the curvature).
        Spectral deviations are made orthogonal to both :math:`u` and
        :math:`q` (`remove_degeneracy_of_spectral_deviations`), so they carry
        neither slope nor curvature and stay zero at the reference frequency.

        Parameters
        ----------
        log_frequencies: ArrayLike
            The log of the frequencies (at least three).
        mean: Model
            Mean of the spectral index.
        spectral_scaled_excitations: ScaledExcitations
            Fluctuations of the spectral index applied to the xis.
        curvature_mean: Model
            Mean of the spectral curvature.
        curvature_scaled_excitations: ScaledExcitations
            Fluctuations of the spectral curvature applied to the xis.
        reference_frequency_index: int
            The index of the reference frequency.
        """
        if len(log_frequencies) < 3:
            raise ValueError("A spectral curvature needs at least three frequencies.")
        target_shape = spectral_scaled_excitations.target.shape
        if curvature_scaled_excitations.target.shape != target_shape:
            raise ValueError("spectral index and curvature fluctuations need the same shape.")
        slicing_tuple = (slice(None),) + (None,) * len(target_shape)

        u = jnp.asarray(log_frequencies - log_frequencies[reference_frequency_index])
        q = u**2 - jnp.sum(u**3) / jnp.sum(u**2) * u
        self._relative_log_frequencies = u[slicing_tuple]
        self._curvature_basis = q[slicing_tuple]
        self._denominator = 1 / jnp.sum(u**2)
        self._curvature_denominator = 1 / jnp.sum(q**2)

        self._mean = mean
        self.spectral_scaled_excitations = spectral_scaled_excitations
        self._curvature_mean = curvature_mean
        self.curvature_scaled_excitations = curvature_scaled_excitations

        domain = reduce(
            lambda a, b: a | b,
            [
                (m.domain.tree if isinstance(m.domain, Vector) else m.domain)
                for m in (mean, spectral_scaled_excitations, curvature_mean, curvature_scaled_excitations)
            ],
        )
        super().__init__(domain=domain)

    @property
    def relative_log_frequencies(self):
        return self._relative_log_frequencies

    @property
    def curvature_basis(self):
        """:math:`q(u)` at the model frequencies, shaped like `relative_log_frequencies`."""
        return self._curvature_basis

    def curvature_basis_at(self, relative_log_frequency):
        """:math:`q(u)` at arbitrary relative log frequencies (same orthogonalisation constant)."""
        u = self.relative_log_frequencies.reshape(-1)
        return relative_log_frequency**2 - jnp.sum(u**3) / jnp.sum(u**2) * relative_log_frequency

    def mean(self, p) -> list[ArrayLike]:
        """Means of the spectral index and the spectral curvature."""
        return [self._mean(p), self._curvature_mean(p)]

    def fluctuations(self, p) -> list[ArrayLike]:
        """Scaled excitations of the spectral index and the spectral curvature."""
        return [self.spectral_scaled_excitations(p), self.curvature_scaled_excitations(p)]

    def fluctuations_with_frequencies(self, p) -> ArrayLike:
        return (
            self.spectral_scaled_excitations(p) * self.relative_log_frequencies
            + self.curvature_scaled_excitations(p) * self.curvature_basis
        )

    def mean_with_frequencies(self, p) -> ArrayLike:
        return self._mean(p) * self.relative_log_frequencies + self._curvature_mean(p) * self.curvature_basis

    def remove_degeneracy_of_spectral_deviations(
        self, deviations: ArrayLike
    ) -> ArrayLike:
        """Remove the components of the deviations along :math:`u` and :math:`q(u)`."""
        dev_slope = jnp.sum(deviations * self.relative_log_frequencies, axis=0) * self._denominator
        dev_curv = jnp.sum(deviations * self.curvature_basis, axis=0) * self._curvature_denominator
        return deviations - dev_slope * self.relative_log_frequencies - dev_curv * self.curvature_basis

    def __call__(self, p):
        # NOTE : only needed for instantiation of the model (as SpectralIndex)
        return None


class SpectralPolynomial(HarmonicLogSpectralBehavior):
    def __init__(
        self,
        log_frequencies: ArrayLike,
        means: list[Model],
        fluctuations: list[Model],
        reference_frequency_index: int,
    ):
        """Polynomial spectral behavior. The spectral index is a special case
        of this model.

        .. math::
            \\log F(\\nu, A, B, ...) = A^{spec.}
            * \\left\\{ \\mathrm{fluc.}^A \\xi^A \\nu + \\mathrm{fluc.}^B \\xi^B \\nu^2 + ... \\right\\}
            + \\mathrm{mean}^A \\nu + \\mathrm{mean}^B \\nu^2 + ...

        Parameters
        ----------
        log_frequencies: ArrayLike
            The log of the frequencies. The relative frequencies are calculated
            internally. See `reference_frequency_index`.
        means: Model
            The means of the polynomial model. See equation above.
        fluctuations: Model
            The fluctuations, already applied to the xis. Hence, the
            combination of \\mathrm{fluc.}^A \\xi^A.
        reference_frequency_index: int
            The index of the reference frequency. Used in order to calculate
            the relative frequencies.
        """

        # NOTE : The sizes between the means and the fluctuations must match.
        assert isinstance(means, list)
        assert isinstance(fluctuations, list)
        assert len(means) == len(fluctuations)
        fluctuations_target_shape = fluctuations[0].target.shape
        for fluc in fluctuations:
            assert fluc.target.shape == fluctuations_target_shape

        # NOTE : The model assumes that spectral axis is fixed to be the first
        # index of the output model. The slicing tuple matches the shape of the
        # relative_log_frequencies using the numpy slicing convention.
        slicing_tuple = (slice(None),) + (None,) * len(fluctuations_target_shape)
        self._relative_log_frequencies = (
            log_frequencies - log_frequencies[reference_frequency_index]
        )[slicing_tuple]

        self._means = means
        self._fluctuations = fluctuations
        # self._denominator = 1 / jnp.sum(self.relative_log_frequencies**2)

        domain = reduce(
            lambda a, b: a | b,
            [
                (m.domain.tree if isinstance(m.domain, Vector) else m.domain)
                for m in self._means + self._fluctuations
            ],
        )

        super().__init__(domain=domain)

    @property
    def relative_log_frequencies(self):
        return self._relative_log_frequencies

    def mean(self, p) -> ArrayLike:
        return [m(p) for m in self._means]

    def fluctuations(self, p) -> ArrayLike:
        return [f(p) for f in self._fluctuations]

    def fluctuations_with_frequencies(self, p) -> ArrayLike:
        """Implements the fluctuations part. See above.

        .. math::
            \\mathrm{fluc.}^A\\xi^A\\nu + \\mathrm{fluc.}^B\\xi^B\\nu^2 + ...
        """

        values = jnp.array(
            [
                f(p) * self.relative_log_frequencies ** (i + 1)
                for i, f in enumerate(self._fluctuations)
            ]
        )

        # NOTE: The 0th-axis contains the polynomial values, hence the sum
        # over 0th-axis returns the polynomial value at the different log-
        # frequencies.
        return jnp.sum(values, axis=0)

    def mean_with_frequencies(self, p) -> ArrayLike:
        """Implements the means part. See above.

        .. math::
            \\mathrm{mean}^A \\nu + \\mathrm{mean}^B \\nu^2 + ...
        """

        values = jnp.array(
            [
                f(p) * self.relative_log_frequencies ** (i + 1)
                for i, f in enumerate(self._means)
            ]
        )

        return jnp.sum(values, axis=0)

    def remove_degeneracy_of_spectral_deviations(
        self, deviations: ArrayLike
    ) -> ArrayLike:
        # TODO :: This needs to be implemented !!!

        raise NotImplementedError

        dev_slope = (
            jnp.sum(deviations * self.relative_log_frequencies, axis=0)
            * self._denominator
        )
        return deviations - dev_slope * self.relative_log_frequencies

    def __call__(self, p):
        # NOTE : only needed for instantiation of the model. However, maybe
        # there is a better way !!!
        # Since the apply is not needed
        return None
