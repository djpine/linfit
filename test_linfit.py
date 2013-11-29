""" Unit tests for linfit
Author: David Pine
November 2013
"""
from linfit import linfit
from numpy import polyfit
import numpy as np
from scipy.linalg import lstsq
import unittest
import timeit

EPS1 = 1.0e-13
EPS2 = 1.0e-10

setup = """
from linfit import linfit
import numpy as np
from scipy.linalg import lstsq
from scipy.stats import linregress
def randomData(xmax, npts):
    x = np.random.uniform(-xmax, xmax, npts)
    scale = np.sqrt(xmax)
    a, b = scale * (np.random.rand(2)-0.5)
    y = a*x + b + a * scale * np.random.randn(npts)
    dy = a * scale * (1.0 + np.random.rand(npts))
    wts = 1./(dy*dy)
    return x, y, dy, wts
x, y, dy, wts = randomData(100., npts)
"""

def randomData(xmax, npts):
    x = np.random.uniform(-xmax, xmax, npts)
    scale = np.sqrt(xmax)
    a, b = scale * (np.random.rand(2)-0.5)
    y = a*x + b + a * scale * np.random.randn(npts)
    dy = a * scale * (1.0 + np.random.rand(npts))
    return x, y, dy

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

class Testlinfitear_BadInputs(unittest.TestCase):

    def test_arraySizeTooSmall(self):
        # check that TypeError is raised for input arrays with length 2
        x = np.array([0, 1])
        y = np.array([2, 3])
        self.assertRaises(TypeError, linfit, x, y)

    def test_arraySizeUnequalxy(self):
        # check that TypeError is raised if input arrays have unequal length
        x = np.array([0, 1, 2, 3])
        y = np.array([2, 3, 4])
        self.assertRaises(TypeError, linfit, x, y)

    def test_arraySizeUnequalydy(self):
        # check that TypeError is raised if y & dy arrays have unequal length
        x = np.array([0, 1, 2, 3])
        y = np.array([2, 3, 4, 5])
        dy = np.array([2, 3, 4])
        self.assertRaises(TypeError, linfit, x, y, sigmay=dy)

class Testlinfit_PerfectFits(unittest.TestCase):
    
    def test_pfits(self):
        # check that perfect straight lines are fit perfectly
        x = np.random.uniform(-100.0, 100.0, 10)
        a, b = np.random.rand(2)
        y = a*x+b
        fit, cvm, redchisq, residuals = linfit(x, y, cov=True, chisq=True, 
                                               residuals=True)
        dfit = [np.sqrt(cvm[i,i]) for i in range(2)]
        diffa, diffb = np.abs(a-fit[0]), np.abs(b-fit[1])
        self.assertTrue(diffa<EPS1 and diffb<EPS1)
        self.assertTrue(np.abs(dfit[0])<EPS1 and np.abs(dfit[1])<EPS1)
        self.assertTrue(np.abs(cvm[0,1])<EPS1*EPS1)
        self.assertTrue(redchisq<EPS1*EPS1)

class Testlinfit_CompareWithOtherFits(unittest.TestCase):
    # Check that linfit gives the same results as other scipy fitting routines

    def test_polyfitWtInd(self):
        # check that get same answers as NumPy polyfit when relative weighting
        # (only type of weighting currently available in polyfit) is used.
        # Also compare run times.
        npts = 100
        x, y, dy = randomData(100., npts)
        wts = 1./dy
        fit, cvm = linfit(x, y, sigmay=dy, relsigma=True, cov=True)
        pfit, v = polyfit(x, y, 1, w=wts, cov=True)
        diff_fit = np.abs((pfit - fit)/fit).sum()/2.0
        # polyfit uses a nonstandard normalization of the covariance
        # matrix.  It is corrected here for comparison with linfit, which
        # uses the standard definition.
        v_corrected = v * (npts-4.)/(npts-2.)
        diff_cvm = np.abs((v_corrected - cvm)/cvm).sum()/4.0
        self.assertTrue(diff_fit<EPS2)
        self.assertTrue(diff_cvm<EPS2)

    def test_polyfitWtSame(self):
        # check that get same answers as NumPy polyfit when relative weighting
        # (only type of weighting currently available in polyfit) is used.
        npts = 100
        x, y, dy = randomData(100., npts)
        dy = np.random.rand(1)
        wts = np.ones(y.size)/dy
        fit, cvm = linfit(x, y, sigmay=dy, relsigma=True, cov=True)
        pfit, v = polyfit(x, y, 1, w=wts, cov=True)
        diff_fit = np.abs((pfit - fit)/fit).sum()/2.0
        # polyfit uses a nonstandard normalization of the covariance
        # matrix.  It is corrected here for comparison with linfit, which
        # uses the standard definition.
        v_corrected = v * (npts-4.)/(npts-2.)
        diff_cvm = np.abs((v_corrected - cvm)/cvm).sum()/4.0
        self.assertTrue(diff_fit<EPS2)
        self.assertTrue(diff_cvm<EPS2)

    def test_linalg_lstsqNoWt(self):
        # check that get same answers as Scipy linalg_lstsq when all weighting
        # is turned off.
        npts = 100
        x, y, dy = randomData(100., npts)
        fit = linfit(x, y)
        X = np.vstack([np.ones(npts), x]).T
        cNoWts, resid, rank, sigma = lstsq(X, y)
        diff_fit = np.abs((np.flipud(cNoWts)-fit)/fit).sum()/2.0
        self.assertTrue(diff_fit<EPS2)

    def test_linalg_lstsqWt(self):
        # check that get same answers as Scipy linalg_lstsq using weighting
        npts = 100
        x, y, dy = randomData(100., npts)
        fit = linfit(x, y, sigmay=dy)
        X = np.vstack([np.ones(npts), x]).T
        A = X/np.array(zip(dy,dy))
        cWts, resid, rank, sigma = lstsq(A, y/dy)
        diff_fit = np.abs((np.flipud(cWts)-fit)/fit).sum()/2.0
        self.assertTrue(diff_fit<EPS2)

class Testlinfit_CompareWithOtherFitsRunTime(unittest.TestCase):
    # These tests compare the times for linfit to fit data sets to other
    # functions in scipy

    def test_polyfitCompareUniformWt(self):
        # Compare runtime of linfit vs polyfit with uniform weighting
        # Relative weighting used since polyfit does not implement absolute weighting
        print('\nCompare linfit to polyfit with unweighted data points')
        nreps = 2
        nruns = 7
        for npts in [10, 100, 1000, 10000, 100000, 1000000]:
            setup1 = "npts="+str(npts)+setup
            timelin = min(timeit.Timer('slope, yint = linfit(x, y)',
                          setup=setup1).repeat(nreps, nruns))
            timepoly = min(timeit.Timer('slope, yint = np.polyfit(x, y, 1)',
                           setup=setup1).repeat(nreps, nruns))
            print("{0:7d} data points: linfit is faster than numpy.polyfit by {1:0.2g} times"
                  .format(npts, timepoly/timelin))

    def test_polyfitCompareIndividualWt(self):
        # Compare runtime of linfit vs polyfit with individually weighted data points
        # Relative weighting used since polyfit does not implement absolute weighting
        print('\nCompare linfit to polyfit with relative individually weighted data points')
        nreps = 2
        nruns = 7
        for npts in [10, 100, 1000, 10000, 100000, 1000000]:
            setup1 = "npts="+str(npts)+setup
            timelin = min(timeit.Timer('fit, cvm = linfit(x, y, sigmay=dy, relsigma=True, cov=True)',
                          setup=setup1).repeat(nreps, nruns))
            timepoly = min(timeit.Timer('pfit, v = np.polyfit(x, y, 1, w=wts, cov=True)',
                           setup=setup1).repeat(nreps, nruns))
            print("{0:7d} data points: linfit is faster than numpy.polyfit by {1:0.2g} times"
                  .format(npts, timepoly/timelin))

    def test_linalg_lstsqCompareUniformWt(self):
        # Compare runtime of linfit vs scipy.linalg.lstsq with no weighting
        print('\nCompare linfit to scipy.linalg.lstsq with unweighted data points')
        nreps = 2
        nruns = 7
        for npts in [10, 100, 1000, 10000, 100000, 1000000]:
            setup1 = "npts="+str(npts)+setup
            timelin = min(timeit.Timer('slope, yint = linfit(x, y)',
                          setup=setup1).repeat(nreps, nruns))
            timelinalg = min(timeit.Timer('X = np.vstack([np.ones(npts), x]).T\ncNoWts, resid, rank, sigma = lstsq(X, y)',
                           setup=setup1).repeat(nreps, nruns))
            print("{0:7d} data points: linfit is faster than scipy.linalg.lstsq by {1:0.2g} times"
                  .format(npts, timelinalg/timelin))

    def test_linalg_lstsqCompareIndividualWt(self):
        # Compare runtime of linfit vs scipy.linalg.lstsq with individual weighting
        print('\nCompare linfit to scipy.linalg.lstsq with relative individually weighted data points')
        nreps = 2
        nruns = 7
        for npts in [10, 100, 1000, 10000]:
            setup1 = "npts="+str(npts)+setup
            timelin = min(timeit.Timer('fit, cvm = linfit(x, y, sigmay=dy, relsigma=True, cov=True)',
                          setup=setup1).repeat(nreps, nruns))
            stmt='X = np.vstack([np.ones(npts), x]).T\nA = X/np.array(zip(dy,dy))\ncWts, resid, rank, sigma = lstsq(A, y/dy)'
            timelinalg = min(timeit.Timer(stmt,
                           setup=setup1).repeat(nreps, nruns))
            print("{0:7d} data points: linfit is faster than scipy.linalg.lstsq by {1:0.3g} times"
                  .format(npts, timelinalg/timelin))

    def test_linregressCompareNoWt(self):
        # Compare runtime of linfit with scipy.stats.linregress
        print('\nCompare linfit to scipy.stats.linregress with unweighted data points')
        nreps = 2
        nruns = 7
        for npts in [10, 100, 1000, 10000, 100000, 1000000]:
            setup1 = "npts="+str(npts)+setup
            timelin = min(timeit.Timer('fit = linfit(x, y)',
                          setup=setup1).repeat(nreps, nruns))
            timelinreg = min(timeit.Timer('slope, intercept, r_value, p_value, std_err = linregress(x, y)',
                             setup=setup1).repeat(nreps, nruns))
            print("{0:7d} data points: linfit is faster than scipy.stats.linregress by {1:0.2g} times"
                  .format(npts, timelinreg/timelin))
            
            
if __name__ == '__main__':
    import sys
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout))
