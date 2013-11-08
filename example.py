import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # for unequal plot boxes
from linfit import linfit

# data set for linear fitting 
x = np.array([2.3, 4.7, 7.1, 9.6, 11.7, 14.1, 16.4, 18.8, 21.1, 23.0])
y = np.array([-25., 3., 114., 110., 234., 304., 271., 322., 446., 397.])
sigmay = np.array([15., 30., 34., 37., 40., 50., 38., 28., 47., 30.])

# Fit linear data set with weighting
fit, cvm, redchisq, residuals = linfit(x, y, sigmay, cov=True,
                                        relsigma=False, residuals=True)
dfit = [np.sqrt(cvm[i,i]) for i in range(2)]

# Open figure window for plotting data with linear fit
fig1 = plt.figure(1, figsize=(8,8))
gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 6])

# Bottom plot: data and fit
ax1 = fig1.add_subplot(gs[1])

# Plot data with error bars on top of fit
ax1.errorbar(x, y, yerr = sigmay, ecolor="black", fmt="ro", ms=5)

# Plot fit (behind data)
endt = 0.05 * (x.max()-x.min())
tFit = np.array([x.min()-endt, x.max()+endt])
vFit = fit[0]*tFit + fit[1]
ax1.plot(tFit, vFit, "-b", zorder=-1)

# Print out results of fit on plot
ax1.text(0.05, 0.9,
    u"slope = {0:0.1f} \xb1 {1:0.1f}".format(fit[0], dfit[0]),
    ha="left", va="center", transform = ax1.transAxes)
ax1.text(0.05, 0.83,
    u"y-intercept = {0:0.1f} \xb1 {1:0.1f}".format(fit[1], dfit[1]),
    ha="left", va="center", transform = ax1.transAxes)
ax1.text(0.05, 0.76,
    "redchisq = {0:0.2f}".format(redchisq),
    ha="left", va="center", transform = ax1.transAxes)
ax1.text(0.05, 0.69,
    "r = {0:0.2f}".format(cvm[0,1]/(dfit[0]*dfit[1])),
    ha="left", va="center", transform = ax1.transAxes)

# Label axes
ax1.set_xlabel("time")
ax1.set_ylabel("velocity")

# Top plot: residuals
ax2 = fig1.add_subplot(gs[0])
ax2.axhline(color="gray", zorder = -1)
ax2.errorbar(x, residuals, yerr = sigmay, ecolor="black", fmt="ro", ms=5)
ax2.set_ylabel("residuals")
ax2.set_ylim(-100, 150)
ax2.set_yticks((-100, 0, 100))

plt.show()
plt.savefig("example.png")
