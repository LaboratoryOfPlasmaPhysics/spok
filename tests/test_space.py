#!/usr/bin/env python

"""Tests for `space` package."""


import unittest
from ddt import data, unpack, ddt
from space.models.planetary import Magnetosheath
import numpy as np
from space import space

@ddt
class TestMagnetosheath(unittest.TestCase):
    """Tests for `space` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.typical_msh = Magnetosheath(magnetopause="mp_shue1998", bow_shock="bs_jelinek2012")
        theta = np.arange(0, np.pi/2, 0.01*np.pi)
        self.typical_mp_sph, self.typical_bs_sph  = self.typical_msh.boundaries(theta, 0, coord_sys = "spherical")
        self.typical_mp_cart, self.typical_bs_cart = self.typical_msh.boundaries(theta, 0, coord_sys = "cartesian")

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_magnetosheath_can_be_constructed(self):
        """Test something."""
        self.assertIsNotNone(self.typical_msh)



    @data(("bs_shue1998", "mp_formisano1979"), #both wrong
          ("mp_shue1998", "mp_formisano1979")) # bow shock wrong
    @unpack
    def test_msh_invalid_boundaries_raise(self, mp, bs):
        """Test something."""
        self.assertRaises(ValueError, Magnetosheath, magnetopause=mp,
                                        bow_shock=bs)


    @data("mp_formisano1979", "mp_shue1998")
    def test_mp_nose_are_in_expected_dayside_region(self, mp):
        msh = Magnetosheath(magnetopause=mp, bow_shock="bs_formisano1979")
        x,y,z = msh.magnetopause(0,0)
        self.assertGreater(x, 5)
        self.assertLess(x, 15)


    @data("bs_formisano1979", "bs_jerab2005")
    def test_mp_nose_are_in_expected_dayside_region(self, bs):
        msh = Magnetosheath(magnetopause="mp_formisano1979", bow_shock=bs)
        x,y,z = msh.bow_shock(0,0)
        self.assertGreater(x, 7)
        self.assertLess(x, 30)


    def test_spherical_and_cartesian_are_consistent(self):
        for model_sph, model_cart in zip((self.typical_mp_sph, self.typical_bs_sph),
                                         (self.typical_mp_cart, self.typical_bs_cart)):
            rfromcart = np.sqrt(model_cart[0]**2 + model_cart[1]**2+ model_cart[2]**2)
            np.testing.assert_allclose(rfromcart, model_sph[0], atol=1e-12)


    def test_msh_return_mp_and_bs(self):
        rmp = self.typical_mp_sph[0]
        rbs = self.typical_bs_sph[0]
        np.testing.assert_array_less(rmp, rbs)


    def test_non_regression(self):
        pass
        # read expected values in pickle file
        # run code for same params
        # check np.testing.allclose(actual, expected)
