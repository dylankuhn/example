import numpy as np
import numpy.testing as npt

from mycosmo.cosmology import hubble, critical_density


class TestCosmology:
    fid_cosmo = {
        "H0": 70,
        "omega_m_0": 0.3,
        "omega_k_0": 0.0,
        "omega_lambda_0": 0.7,
    }
    H_tolerance = 0.01
    d_tolerance = 1e-27
    z_range = np.array([0.0, 0.5, 1.0])
    H_expect = np.array([70, 91.60, 123.24])
    d_expect = np.array([9.203859495267889e-27, 1.576160938564626e-26, 2.853196443533045e-26])

    def test_hubble(self):
        H_vals = hubble(self.z_range, self.fid_cosmo)

        npt.assert_allclose(
            H_vals,
            self.H_expect,
            atol=self.H_tolerance,
            err_msg=(
                "The H(z) differs from expected values by more than "
                f"{self.H_tolerance} decimal places."
            ),
        )

    def test_critical_density(self):
        d_vals = critical_density(self.z_range, self.fid_cosmo)

        npt.assert_allclose(
            d_vals,
            self.d_expect,
            atol=self.d_tolerance,
            err_msg=(
                "The critical density differs from expected values by more than "
                f"{self.d_tolerance} decimal places."
            ),
        )
