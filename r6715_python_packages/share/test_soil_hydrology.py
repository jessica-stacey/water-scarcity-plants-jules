
import unittest
import numpy as np

import soil_hydrology


class TestDarcy(unittest.TestCase):
    def test_against_kgo_from_jules(self):
        # using kgo from JULES vn4.8

        b = np.array([6.224, 6.224])
        sathh = np.array([0.1971, 0.1971])
        ks = 0.005133
        dz1 = 0.1
        dz2 = 0.25 
        sthu1 = 0.63400000
        sthu2 = 0.63399994

        l_dpsids_dsdz = False
       
        wflux_kgo = 4.49832532E-06
        dwflux_dsthu1_kgo = 9.26437671E-04
        dwflux_dsthu2_kgo = -8.16832297E-04
        rtol = 5e-06 # can't distinguish between l_dpsids_dsdz at this rtol

        wflux, dwflux_dsthu1, dwflux_dsthu2 = soil_hydrology.darcy(
            b, dz1, dz2, ks, sathh, sthu1, sthu2, l_dpsids_dsdz=l_dpsids_dsdz)
 
        np.testing.assert_allclose(wflux, wflux_kgo, rtol=rtol)
        np.testing.assert_allclose(dwflux_dsthu1, dwflux_dsthu1_kgo, rtol=rtol)
        np.testing.assert_allclose(dwflux_dsthu2, dwflux_dsthu2_kgo, rtol=rtol)

        l_dpsids_dsdz = True
       
        wflux_kgo = 4.49832532E-06
        dwflux_dsthu1_kgo = 9.26437613E-04
        dwflux_dsthu2_kgo = -8.16832355E-04
        
        wflux, dwflux_dsthu1, dwflux_dsthu2 = soil_hydrology.darcy(
            b, dz1, dz2, ks, sathh, sthu1, sthu2, l_dpsids_dsdz=l_dpsids_dsdz)

        np.testing.assert_allclose(wflux, wflux_kgo, rtol=rtol)
        np.testing.assert_allclose(dwflux_dsthu1, dwflux_dsthu1_kgo, rtol=rtol)
        np.testing.assert_allclose(dwflux_dsthu2, dwflux_dsthu2_kgo, rtol=rtol)


class TestSoilHydrology(unittest.TestCase):
    def test_against_kgo_from_jules(self):
        # using kgo from JULES vn4.8

        dz = np.array([0.1,0.25,0.35,0.35,1.5])

        n = len(dz)
        v_sat = np.full((n,), 0.4304)
        bexp = np.full((n,), 6.224)
        ksz = np.full((n,), 0.005133)
        sathh = np.full((n,), 0.1971)

        l_dpsids_dsdz = True

        timestep = 3600

        sthu_in = np.array([0.63400000, 0.63399994, 1.5, 0.63400000, 0.90850002])
        fw = 0.0
        ext = np.array([6.15025385E-07, 0.0, 0.0, 0.0, 0.0])

        rtol = 5e-06 # can distinguish between l_soil_sat_down at this rtol

        l_soil_sat_down = True

        sthu, dsmcl, wflux_dummy = soil_hydrology.soil_hydrology(
            sthu_in, bexp=bexp, ksz=ksz, dz=dz, v_sat=v_sat, ext=ext, 
            timestep=timestep, sathh=sathh, fw=fw, 
            l_soil_sat_down=l_soil_sat_down, l_dpsids_dsdz=l_dpsids_dsdz)

        sthu_kgo = np.array([0.62488359, 0.49648827, 0.64514786, 0.66165608, 0.99065584]) 
        dsmcl_kgo = np.array([-0.39236963, -14.796257, -128.77492, 4.1661181, 53.039818])  

        np.testing.assert_allclose(sthu, sthu_kgo, rtol=rtol)
        np.testing.assert_allclose(dsmcl, dsmcl_kgo, rtol=rtol) 

        l_soil_sat_down = False

        sthu, dsmcl, wflux_dummy = soil_hydrology.soil_hydrology(
            sthu_in, bexp=bexp, ksz=ksz, dz=dz, v_sat=v_sat, ext=ext, 
            timestep=timestep, sathh=sathh, fw=fw, 
            l_soil_sat_down=l_soil_sat_down, l_dpsids_dsdz=l_dpsids_dsdz)

        sthu_kgo = np.array([0.96551359, 0.50523710, 0.68736666, 0.66167730, 0.90238619]) 
        dsmcl_kgo = np.array([14.268346, -13.854879, -122.41508, 4.1693020, -3.9471076])

        np.testing.assert_allclose(sthu, sthu_kgo, rtol=rtol)
        np.testing.assert_allclose(dsmcl, dsmcl_kgo, rtol=rtol)


if __name__ == '__main__':
    unittest.main()
