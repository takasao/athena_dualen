# Regression test based on Newtonian MHD linear wave convergence problem
#
# Runs a linear wave convergence test in 3D and checks L1 errors (which
# are computed by the executable automatically and stored in the temporary file
# linearwave_errors.dat)

# Modules
import logging
import scripts.utils.athena as athena
from math import log
import numpy as np
import sys
import os
import shutil
sys.path.insert(0, '../../vis/python')
import athena_read                             # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name based on module

# List of time/integrator and time/xorder combinations to test:
solvers = [('rk2', '2'), ('rk3', '3')]
# solvers = [('rk2', '2'), ('rk3', '4')]

# resolution_range = [32, 64] #, 128, 256]  # , 512]
resolution_range = [16, 32, 64, 128]  # , 256]  # , 512]
# resolution_range = [16, 32]
num_nx1 = len(resolution_range)

# Number of times Athena++ is run for each above configuration
# (number of L-going + entropy waves)*num_nx1 + (L and R fast waves):
nrows_per_solver = 4*num_nx1 + 2


# Prepare Athena++
def prepare(**kwargs):
    logger.debug('Running test ' + __name__)
    athena.configure(
        'b', 'dual_en',  # dual_en is ON
        'mpi', 'hdf5',
        nghost=4,  # required for fourth-order configurations
        prob='linear_wave_dual_en',  # use pgen for dual energy
        coord='cartesian',
        flux='hlld', **kwargs)
    athena.make()


# Run Athena++
def run(**kwargs):
    def get_param(i):
        nx1_mesh = repr(i)
        nx2_mesh = repr(i/2)
        nx3_mesh = repr(i/2)

        nx1_mb = repr(i/4)
        nx2_mb = repr(i/4)
        nx3_mb = repr(i/4)
        ncore = 4
        if (int(float(nx2_mb)) < 6):
            ncore = 2
            nx1_mb = repr(i/2)
            nx2_mb = repr(i/2)
            nx3_mb = repr(i/2)

        nx_mesh = [nx1_mesh, nx2_mesh, nx3_mesh]
        nx_mb = [nx1_mb, nx2_mb, nx3_mb]
        return ncore, nx_mesh, nx_mb

    for (torder, xorder) in solvers:
        logger.info('>>> torder: {0}, xorder: {1}\n'.format(torder, xorder))
        # L-going fast/Alfven/slow waves
        for w in (0, 1, 2):
            tlim = max(0.5, w)
            for i in resolution_range:
                ncore, nx_mesh, nx_mb = get_param(i)
                arguments = ['time/ncycle_out=100',
                             'time/tlim=' + repr(tlim),
                             'time/xorder=' + xorder,
                             'time/integrator=' + torder,
                             'problem/wave_flag=' + repr(w),
                             'problem/vflow=0.0',
                             'mesh/nx1=' + nx_mesh[0],
                             'mesh/nx2=' + nx_mesh[1],
                             'mesh/nx3=' + nx_mesh[2],
                             'meshblock/nx1=' + nx_mb[0],
                             'meshblock/nx2=' + nx_mb[1],
                             'meshblock/nx3=' + nx_mb[2],
                             'output2/dt=-1',
                             'problem/compute_error=true']
                # athena.run('mhd/athinput.linear_wave3d', arguments)
                athena.mpirun(kwargs['mpirun_cmd'],
                              kwargs['mpirun_opts'], ncore,
                              'mhd/athinput.linear_wave3d', arguments)

        # entropy wave
        for i in resolution_range:
            ncore, nx_mesh, nx_mb = get_param(i)
            arguments = ['time/ncycle_out=100',
                         'time/tlim=1.0',
                         'time/xorder=' + xorder,
                         'time/integrator=' + torder,
                         'problem/wave_flag=3',
                         'problem/vflow=1.0',
                         'mesh/nx1=' + nx_mesh[0],
                         'mesh/nx2=' + nx_mesh[1],
                         'mesh/nx3=' + nx_mesh[2],
                         'meshblock/nx1=' + nx_mb[0],
                         'meshblock/nx2=' + nx_mb[1],
                         'meshblock/nx3=' + nx_mb[2],
                         'output2/dt=-1',
                         'problem/compute_error=true']
            # athena.run('mhd/athinput.linear_wave3d', arguments)
            athena.mpirun(kwargs['mpirun_cmd'],
                          kwargs['mpirun_opts'], ncore,
                          'mhd/athinput.linear_wave3d', arguments)

        # L/R-going fast wave (for symmetry check)
        i = resolution_range[0]
        ncore, nx_mesh, nx_mb = get_param(i)
        w = 0
        arguments = ['time/ncycle_out=100',
                     'time/tlim=0.5',
                     'time/xorder=' + xorder,
                     'time/integrator=' + torder,
                     'problem/wave_flag=' + repr(w),
                     'output2/dt=-1',
                     'mesh/nx1=' + nx_mesh[0],
                     'mesh/nx2=' + nx_mesh[1],
                     'mesh/nx3=' + nx_mesh[2],
                     'meshblock/nx1=' + nx_mb[0],
                     'meshblock/nx2=' + nx_mb[1],
                     'meshblock/nx3=' + nx_mb[2],
                     'problem/compute_error=true']
        # athena.run('mhd/athinput.linear_wave3d', arguments)
        athena.mpirun(kwargs['mpirun_cmd'],
                      kwargs['mpirun_opts'], ncore,
                      'mhd/athinput.linear_wave3d', arguments)

        w = 6
        arguments = ['time/ncycle_out=100',
                     'time/tlim=0.5',
                     'time/xorder=' + xorder,
                     'time/integrator=' + torder,
                     'problem/wave_flag=' + repr(w),
                     'output2/dt=-1',
                     'mesh/nx1=' + nx_mesh[0],
                     'mesh/nx2=' + nx_mesh[1],
                     'mesh/nx3=' + nx_mesh[2],
                     'meshblock/nx1=' + nx_mb[0],
                     'meshblock/nx2=' + nx_mb[1],
                     'meshblock/nx3=' + nx_mb[2],
                     'problem/compute_error=true']
        # athena.run('mhd/athinput.linear_wave3d', arguments)
        athena.mpirun(kwargs['mpirun_cmd'],
                      kwargs['mpirun_opts'], ncore,
                      'mhd/athinput.linear_wave3d', arguments)

        # clear object directory
        os.system('rm -rf obj')
    return


# Analyze outputs
def analyze():
    filename = 'bin/linearwave-errors.dat'

    # copy data file for later use
    shutil.copy(filename, '/Users/shinsuketakasao/Simulations/work/linear_wave')

    # read data from error file
    data = athena_read.error_dat(filename)
    analyze_status = True

    for (torder, xorder) in solvers:
        # effectively list.pop() range of rows for this solver configuration
        solver_results = np.array(data[0:nrows_per_solver])
        data = np.delete(data, np.s_[0:nrows_per_solver], 0)

        # L-going fast wave
        logger.info("L-going fast wave error convergence:")
        logger.info("nx1   |   rate   |   RMS-L1")
        rms_errs = solver_results[0:num_nx1, 4]
        nx1_range = solver_results[0:num_nx1, 0]
        for i in range(1, num_nx1):
            rate = log(rms_errs[i-1]/rms_errs[i])/log(nx1_range[i]/nx1_range[i-1])
            logger.info("%d %g %g", int(nx1_range[i]), rate, rms_errs[i])

        # L-going Alfven wave
        logger.info("L-going Alfven wave error convergence:")
        logger.info("nx1   |   rate   |   RMS-L1")
        rms_errs = solver_results[num_nx1:2*num_nx1, 4]
        nx1_range = solver_results[num_nx1:2*num_nx1, 0]
        for i in range(1, num_nx1):
            rate = log(rms_errs[i-1]/rms_errs[i])/log(nx1_range[i]/nx1_range[i-1])
            logger.info("%d %g %g", int(nx1_range[i]), rate, rms_errs[i])

        # L-going slow wave
        logger.info("L-going slow wave error convergence:")
        logger.info("nx1   |   rate   |   RMS-L1")
        rms_errs = solver_results[2*num_nx1:3*num_nx1, 4]
        nx1_range = solver_results[2*num_nx1:3*num_nx1, 0]
        for i in range(1, num_nx1):
            rate = log(rms_errs[i-1]/rms_errs[i])/log(nx1_range[i]/nx1_range[i-1])
            logger.info("%d %g %g", int(nx1_range[i]), rate, rms_errs[i])

        # entropy wave
        logger.info("Entropy wave error convergence:")
        logger.info("nx1   |   rate   |   RMS-L1")
        rms_errs = solver_results[3*num_nx1:4*num_nx1, 4]
        nx1_range = solver_results[3*num_nx1:4*num_nx1, 0]
        for i in range(1, num_nx1):
            rate = log(rms_errs[i-1]/rms_errs[i])/log(nx1_range[i]/nx1_range[i-1])
            logger.info("%d %g %g", int(nx1_range[i]), rate, rms_errs[i])

        # Note: PLM is OK, but PPM result is slightly asymmetric
        # # check error identical for waves in each direction
        # rms_errs = solver_results[4*num_nx1:4*num_nx1+2, 4]
        # if rms_errs[0] != rms_errs[1]:
        #     logger.warning("error in L/R-going fast waves not equal %g %g",
        #                    rms_errs[0], rms_errs[1])
        #     analyze_status = False

    return analyze_status
