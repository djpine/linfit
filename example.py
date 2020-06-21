from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # for unequal plot boxes
from linfit import linfit

# data set for linear fitting 
x = np.array([2.3, 4.7, 7.1, 9.6, 11.7, 14.1, 16.4, 18.8, 21.1, 23.0])
y = np.array([-25., 3., 110., 110., 230., 300., 270., 320., 450., 400.])
sigmay = np.array([15., 30., 30., 40., 40., 50., 40., 30., 50., 30.])

# Fit linear data set with weighting
fit, cvm, info = linfit(x, y, sigmay, relsigma=False, return_all=True)
dfit = np.sqrt(np.diag(cvm))  # uncertainty estimates for fitting parameters

# Open figure window for plotting data with linear fit
fig = plt.figure(1, figsize=(8, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 6])

# Bottom plot: data and fit
ax1 = fig.add_subplot(gs[1])

# Plot data with error bars
ax1.errorbar(x, y, yerr=sigmay, ecolor='k', mec='k', fmt='oC3', ms=6)

# Plot fit (behind data)
endx = 0.05 * (x.max() - x.min())
xFit = np.array([x.min() - endx, x.max() + endx])
yFit = fit[0] * xFit + fit[1]
ax1.plot(xFit, yFit, '-', zorder=-1)

# Print out results of fit on plot
ax1.text(0.05, 0.9,  # slope of fit
         u'slope = {0:0.1f} \xb1 {1:0.1f}'.format(fit[0], dfit[0]),
         ha='left', va='center', transform=ax1.transAxes)
ax1.text(0.05, 0.83,  # y-intercept of fit
         u'y-intercept = {0:0.1f} \xb1 {1:0.1f}'.format(fit[1], dfit[1]),
         ha='left', va='center', transform=ax1.transAxes)
ax1.text(0.05, 0.76,  # reduced chi-squared of fit
         'redchisq = {0:0.2f}'.format(info.rchisq),
         ha='left', va='center', transform=ax1.transAxes)
ax1.text(0.05, 0.69,  # correlation coefficient of fitted slope & y-intercept
         'rcov = {0:0.2f}'.format(cvm[0, 1] / (dfit[0] * dfit[1])),
         ha='left', va='center', transform=ax1.transAxes)

# Label axes
ax1.set_xlabel('time')
ax1.set_ylabel('velocity')

# Top plot: residuals
ax2 = fig.add_subplot(gs[0])
ax2.axhline(color='gray', lw=0.5, zorder=-1)
ax2.errorbar(x, info.resids, yerr=sigmay, ecolor='k', mec='k', fmt='oC3', ms=6)
ax2.set_ylabel('residuals')
ax2.set_ylim(-100, 150)
ax2.set_yticks((-100, 0, 100))

plt.savefig('example.png')
plt.show()
