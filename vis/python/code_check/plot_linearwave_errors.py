import argparse
import athena_read
import matplotlib.pyplot as plt
from math import log
import numpy as np


def get_errors(res, nwaves, num_nx1):
    err = []
    for i in range(nwaves):
        i1 = i * num_nx1
        i2 = (i + 1) * num_nx1
        # 5th column contains errors (fast, Alfven, slow, and entropy modes)
        err.append(res[i1:i2,5])
    return err


def make_plot(resol, data1, data2, imgname):
    y1_f = data1[0] # fast
    y1_A = data1[1] # Alfven
    y1_s = data1[2] # slow
    y1_e = data1[3] # entropy

    y2_f = data2[0] # fast
    y2_A = data2[1] # Alfven
    y2_s = data2[2] # slow
    y2_e = data2[3] # entropy

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    color = 'b'; marker='^'
    # Plot the data in each subplot
    axs[0, 0].plot(resol, y1_f, color=color, marker=marker)
    axs[0, 1].plot(resol, y1_A, color=color, marker=marker)
    axs[1, 0].plot(resol, y1_s, color=color, marker=marker)
    axs[1, 1].plot(resol, y1_e, color=color, marker=marker, label='RK2 + PLM')

    color = 'orange'; marker='s'
    # Plot the data in each subplot
    axs[0, 0].plot(resol, y2_f, color=color, marker=marker)
    axs[0, 1].plot(resol, y2_A, color=color, marker=marker)
    axs[1, 0].plot(resol, y2_s, color=color, marker=marker)
    axs[1, 1].plot(resol, y2_e, color=color, marker=marker, label='RK3 + PPM')

    xlim = [10, 1e3]
    ylim = [1e-12, 1e-6]
    for j in range(2):
        for i in range(2):
            axs[j, i].set_xscale('log')
            axs[j, i].set_yscale('log')
            axs[j, i].set_xlim(xlim)
            axs[j, i].set_ylim(ylim)
            axs[j, i].set_xlabel(r'$N_{x_1}$')

            if (i==0 and j==0):
                coef_2nd = y1_f[0]*resol[0]**2
                coef_3rd = y2_f[0]*resol[0]**3
                axs[j, i].set_ylabel(r'RMS L1 Error (Fast Wave)')
            elif (i==1 and j==0):
                coef_2nd = y1_A[0]*resol[0]**2
                coef_3rd = y2_A[0]*resol[0]**3
                axs[j, i].set_ylabel(r'RMS L1 Error (Alfven Wave)')
            elif (i==0 and j==1):
                coef_2nd = y1_s[0]*resol[0]**2
                coef_3rd = y2_s[0]*resol[0]**3
                axs[j, i].set_ylabel(r'RMS L1 Error (Slow Wave)')
            elif (i==1 and j==1):
                coef_2nd = y1_e[0]*resol[0]**2
                coef_3rd = y2_e[0]*resol[0]**3
                axs[j, i].set_ylabel(r'RMS L1 Error (Entropy Wave)')
                # 4th order
                coef_4th = y2_e[0]*resol[0]**4

            axs[j, i].plot(resol, coef_2nd*resol**(-2), linestyle='dashed', color='k',label='2nd')
            axs[j, i].plot(resol, coef_3rd*resol**(-3), linestyle='dotted', color='k',label='3rd')

            if (i==1 and j==1):
                axs[j, i].plot(resol, coef_4th*resol**(-4), linestyle='-.', color='k',label='4th')
                axs[1, 1].legend(loc='lower left')

    # Adjust spacing between subplots
    fig.tight_layout()

    # Save the figure as a PNG file
    print('>>> Saving '+imgname)
    plt.savefig(imgname, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a figure file.')
    # Add the arguments
    parser.add_argument("filename", help="Input data filename")
    parser.add_argument("imgname", help="Output image filename")
    # Parse the arguments
    args = parser.parse_args() # necessary
    filename = args.filename
    imgname = args.imgname

    # #filename = 'linearwave-errors_01.dat'
    # filename = 'linearwave-errors_xorder3_uniform.dat'
    data = athena_read.error_dat(filename)
    # note: last two lines are for symmetry check. 
    # So we do not use them in this script.

    solver1 = ('rk2', '2')
    solver2 = ('rk3', '3')
    solvers = [solver1, solver2]
    nwaves = 4 # number of waves considered

    resolution_range = sorted(set(data[:,0]))
    resolution_range = np.array(resolution_range)
    num_nx1 = len(resolution_range)

    print('>>> Resolution: ',resolution_range)

    # Number of times Athena++ is run for each above configuration:
    nrows_per_solver = nwaves*num_nx1 + 2

    results1 = np.array(data[0:nrows_per_solver])
    results2 = np.array(data[nrows_per_solver:2*nrows_per_solver])

    errors1 = get_errors(results1, nwaves, num_nx1)
    errors2 = get_errors(results2, nwaves, num_nx1)

    make_plot(resolution_range, errors1, errors2, imgname)

# # print convergence rate for quantitative check
# for (torder, xorder) in solvers:
#     # effectively list.pop() range of rows for this solver configuration
#     solver_results = np.array(data[0:nrows_per_solver])
#     data = np.delete(data, np.s_[0:nrows_per_solver], 0)

#     print('>>> torder: ',torder,'xorder: ',xorder)

#     # entropy wave
#     print("Entropy wave error convergence:")
#     print("nx1   |   rate   |   RMS-L1")
#     rms_errs = solver_results[0:num_nx1, 4]
#     nx1_range = solver_results[0:num_nx1, 0]
#     for i in range(1, num_nx1):
#         rate = log(rms_errs[i-1]/rms_errs[i])/log(nx1_range[i]/nx1_range[i-1])
#         print("%d     %g     %g", int(nx1_range[i]), rate, rms_errs[i])

    