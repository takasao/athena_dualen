//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file snr.cpp
//! \brief Problem generator for supernova remnant.
//! Authors: Kazunari Iwasaki, Kazuyuki Sugimura
//!
//! REFERENCE:
//! For cooling and heating function,
//! - Sutherland, R. S., & Dopita, M. A. 1993, ApJS, 88, 253
//! - Koyama, H., & Inutsuka, S.-i. 2002, ApJL, 564, L97
//! - Kim, C.-G., & Ostriker, E. C. 2017, ApJ, 846, 133

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <iomanip>    // std::setprecision
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

namespace {
Real threshold;

//cooling
int ntab = 128;
AthenaArray<Real> logLam, mol, logT0;
Real logT0_min, logT0_max, dlogT0;
Real H_FUV = 2.8e-26; // FUV heating rate
Real Lunit; // unit of length
Real tunit; // unit of time
Real vunit; // unit of velocity
Real rhounit; // unit of density
Real nHunit; // unit of hydrogen nuclei number density
Real Munit;  // unit of mass
Real Eunit; // unit of energy
Real eunit; // unit of energy density
Real Tunit; // unit of temperature
Real coolunit; // unit of coolingrate

Real namb;
Real Tamb;
Real rini;
Real Eini;
Real Mej;
Real gm1; //gamma-1
Real b0;

const Real cgs_Msun = 1.9891e+33; //solar mass [g]
const Real cgs_pc = 3.08568e+18; //parsec [cm]
const Real cgs_Gnewton = 6.6726e-08; //Newton's constant [cm^3/g/s^2]
const Real cgs_mp = 1.67262171e-24; //proton mass [g]
const Real cgs_yr = 31557600.0; //yr [sec]
const Real cgs_kB = 1.3806505e-16; //Boltzmann constant [erg/K]
} // namespace

void MyDualEnSwitchingFunc(Real &ei_nonc,
          const Real &ei_cons, const Real &ekin, const Real &emag);

//cooling
void CoolingFunc(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
    AthenaArray<Real> &cons_scalar);
void SetCoolingTable();
Real InterpolateLambda(Real tem0);
Real MyTimeStep(MeshBlock *pmb);
void DefineSimulationUnits();
void AddThermalEnergyInsideRsn(Mesh *pm, Real xsn, Real ysn, Real zsn);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (DUAL_ENERGY_ENABLED)
    EnrollUserDualEnergySwitchingFunction(MyDualEnSwitchingFunc);

  // cooling
  EnrollUserExplicitSourceFunction(CoolingFunc);
  EnrollUserTimeStepFunction(MyTimeStep);
  DefineSimulationUnits();
  SetCoolingTable();

  // physical parameters
  namb = pin->GetReal("problem", "namb");
  Tamb = pin->GetReal("problem", "Tamb");
  rini = pin->GetReal("problem", "rini");
  Eini = pin->GetReal("problem", "Eini");
  Mej = pin->GetReal("problem", "Mej");
  Real gamma = pin->GetReal("hydro", "gamma"); // read directly from input file
  gm1 = gamma - 1.0; // save gamma-1 as a global variable gm1

  b0 = pin->GetOrAddReal("problem", "b0", 0.0); // in microgauss
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief problem generator for supernova remnant
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; k++) {
    Real z = pcoord->x3v(k);
    for (int j=js; j<=je; j++) {
      Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; i++) {
        Real x = pcoord->x1v(i);
        Real den = namb / nHunit;
        Real vx  = 0.0;
        Real vy  = 0.0;
        Real vz  = 0.0;
        Real prs = Tamb / Tunit * den;

        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = den*vx;
        phydro->u(IM2,k,j,i) = den*vy;
        phydro->u(IM3,k,j,i) = den*vz;
        phydro->u(IEN,k,j,i) = prs/gm1 + 0.5*den*(SQR(vx)+SQR(vy)+SQR(vz));
      }
    }
  }

  // initialize interface B and total energy
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        pfield->b.x1f(k,j,i) = 0.0;
      }
    }
  }
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x2f(k,j,i) = 0.0;
      }
    }
  }
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x3f(k,j,i) = b0;
      }
    }
  }
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IEN,k,j,i) += 0.5*b0*b0;
      }
    }
  }

  // Add mass and energy to drive supernova
  if (lid == pmy_mesh->nblocal -1) {
#ifdef MPI_PARALLEL
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      Real xsn = 0.0; // center of SN
      Real ysn = 0.0; // center of SN
      Real zsn = 0.0; // center of SN

      AddThermalEnergyInsideRsn(pmy_mesh, xsn, ysn, zsn);
  }
}

void MyDualEnSwitchingFunc(Real &ei_nonc,
          const Real &ei_cons, const Real &ekin, const Real &emag) {
  // Synchronize ein with etot using switching function
  ei_nonc = (ei_cons > 0.99*ei_nonc) ? ei_cons : ei_nonc;
  return;
}

void DefineSimulationUnits() {
  Lunit    = cgs_pc;
  tunit    = 1e6 * cgs_yr;
  rhounit  = cgs_mp;
  vunit    = Lunit / tunit;
  Munit    = rhounit * Lunit * Lunit * Lunit;
  Eunit    = Munit * SQR(vunit);
  eunit    = rhounit * SQR(vunit);
  Tunit    = SQR(vunit) * cgs_mp / cgs_kB; // code_velocity^2 -> T/mu in K
  coolunit = eunit / tunit;
  nHunit   = 1.0 / 1.4; // code_density -> nH in cm-3
}

void CoolingFunc(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
    AthenaArray<Real> &cons_scalar) {
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real tem0 = Tunit * prim(IPR,k,j,i) / prim(IDN,k,j,i);
        Real numH = prim(IDN,k,j,i) * nHunit;

        Real logLambda = InterpolateLambda(tem0);
        // FUV heating and radiative cooling
        // FUV heating is turned off when T > 8000K because dusts are destroyed.
        Real dEdt = numH * (H_FUV*0.5*( 1.0 + SIGN(8000.0 - tem0))
                  - numH * std::pow(10,logLambda) ) / coolunit;
        cons(IEN,k,j,i) += dEdt * dt;
      }
    }
  }
}

Real InterpolateLambda(Real tem0) {
  Real tmp = std::min(log10(tem0),logT0_max);
  Real lt  = std::max((tmp-logT0_min) / dlogT0, 0.0);
  int it = std::max(std::min(static_cast<int>(lt),ntab-2), 0); // integer
  Real wt=lt - (Real) it;

  return (logLam(it) * (1.0 - wt) + logLam(it + 1) * wt);
}

Real MyTimeStep(MeshBlock *pmb) {
  Real dt_cool = 1e100;
  const AthenaArray<Real> &prim = pmb->phydro->w;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real tem0 = Tunit * prim(IPR,k,j,i) / prim(IDN,k,j,i);
        Real numH = prim(IDN,k,j,i) * nHunit;

        Real logLambda = InterpolateLambda(tem0);
        // FUV heating and radiative cooling
        // FUV heating is turned off when T > 8000K because dusts are destroyed.
        Real dEdt = numH * (H_FUV * 0.5 * (1.0 + SIGN(8000.0 - tem0))
              - numH * std::pow(10,logLambda)) / coolunit;
        dt_cool = std::min(dt_cool, 0.1 * prim(IPR,k,j,i) / fabs(dEdt));
      }
    }
  }
  return (pmb->pmy_mesh->cfl_number * dt_cool);
}

void AddThermalEnergyInsideRsn(Mesh *pm, Real xsn, Real ysn, Real zsn) {
  Real data[2] = {0.0,0.0}; // data[0] = Msn, data[1] = vol
  Real Rsn = rini * cgs_pc/Lunit;
  Real Rsn2 = Rsn * Rsn;

  // Get volume and total mass within the radius of Rsn
  for (int b=0; b<pm->nblocal; ++b) {
    MeshBlock* pmb = pm->my_blocks(b);
    Coordinates *pcoord = pmb->pcoord;
    int is = pmb->is, ie = pmb->ie;
    int js = pmb->js, je = pmb->je;
    int ks = pmb->ks, ke = pmb->ke;

    // Get the location of SN center
    // uniform mesh is assumed
    Real dx = pcoord->dx1f(pmb->is);
    Real dy = pcoord->dx2f(pmb->js);
    Real dz = pcoord->dx3f(pmb->ks);
    Real dvol = dx * dy * dz;

    int ii  = static_cast<int>( (xsn - pcoord->x1f(is))/dx ) + is;
    int jj  = static_cast<int>( (ysn - pcoord->x2f(js))/dy ) + js;
    int kk  = static_cast<int>( (zsn - pcoord->x3f(ks))/dz ) + ks;

    int fsn = static_cast<int>(Rsn/dx) + 1;

    int iim = std::max(std::min(ii - fsn,ie), is);
    int jjm = std::max(std::min(jj - fsn,je), js);
    int kkm = std::max(std::min(kk - fsn,ke), ks);
    int iip = std::max(std::min(ii + fsn,ie), is);
    int jjp = std::max(std::min(jj + fsn,je), js);
    int kkp = std::max(std::min(kk + fsn,ke), ks);

    Hydro *phydro = pmb->phydro;
    for (int k=kkm; k<=kkp; ++k) {
      Real dz2 = SQR(pcoord->x3v(k) - zsn);
      for (int j=jjm; j<=jjp; ++j) {
        Real dy2 = SQR(pcoord->x2v(j) - ysn);
        for (int i=iim; i<=iip; ++i) {
          if( SQR(pcoord->x1v(i) - xsn) + dy2 + dz2 < Rsn2 ) {
            data[0] += dvol;
            data[1] += phydro->u(IDN,k,j,i) * dvol;
          }
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, data, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // SNR volume
  Real volsn = data[0];

  // SNR mass
  Real Msn = data[1] + Mej * cgs_Msun / Munit;

  //----------------- homogenize and add energy -----------------//
  Real denave = Msn / volsn; // average density
  Real Phot = Eini / Eunit * gm1 / volsn; // pressure defined by SN energy

  for (int b=0; b<pm->nblocal; ++b) {
    MeshBlock* pmb = pm->my_blocks(b);
    Coordinates *pcoord = pmb->pcoord;
    int is = pmb->is, ie = pmb->ie;
    int js = pmb->js, je = pmb->je;
    int ks = pmb->ks, ke = pmb->ke;

    // Get the location of SN center
    // uniform mesh is assumed
    Real dx = pcoord->dx1f(pmb->is);
    Real dy = pcoord->dx2f(pmb->js);
    Real dz = pcoord->dx3f(pmb->ks);
    Real dvol = dx * dy * dz;

    int ii  = static_cast<int>( (xsn - pcoord->x1f(is))/dx ) + is;
    int jj  = static_cast<int>( (ysn - pcoord->x2f(js))/dy ) + js;
    int kk  = static_cast<int>( (zsn - pcoord->x3f(ks))/dz ) + ks;

    int fsn = static_cast<int>(Rsn/dx) + 1;

    int iim = std::max(std::min(ii - fsn,ie+NGHOST), is-NGHOST);
    int jjm = std::max(std::min(jj - fsn,je+NGHOST), js-NGHOST);
    int kkm = std::max(std::min(kk - fsn,ke+NGHOST), ks-NGHOST);
    int iip = std::max(std::min(ii + fsn,ie+NGHOST), is-NGHOST);
    int jjp = std::max(std::min(jj + fsn,je+NGHOST), js-NGHOST);
    int kkp = std::max(std::min(kk + fsn,ke+NGHOST), ks-NGHOST);

    if (pmb->gid == 0) {
      std::cout << ">>> Add thermal energy of "
                << std::scientific << std::setprecision(3) << Eini
                << " erg" << std::endl;
    }

    Hydro *phydro = pmb->phydro;
    Field *pfield = pmb->pfield;
    for (int k=kkm; k<=kkp; ++k) {
      Real dz2 = SQR(pcoord->x3v(k) - zsn);
      for (int j=jjm; j<=jjp; ++j) {
        Real dy2 = SQR(pcoord->x2v(j) - ysn);
#pragma omp simd
        for (int i=iim; i<=iip; ++i) {
          Real dx2 = SQR(pcoord->x1v(i) - xsn);
          Real rad2 = dx2 + dy2 + dz2;
          if (rad2 < Rsn2) {
            Real rad = sqrt(rad2);

            // conservative
            phydro->u(IDN,k,j,i) = denave;
            phydro->u(IEN,k,j,i) = Phot / gm1
              + 0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1)))
                    +SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i)))
                    +SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i))));

            // update primitive to calculate dt_cool later
            phydro->w(IDN,k,j,i) = denave;
            phydro->w(IPR,k,j,i) = Phot;
          }
        }
      }
    }
  }
}

void SetCoolingTable() {
  logT0.NewAthenaArray(ntab);
  logLam.NewAthenaArray(ntab);
  mol.NewAthenaArray(ntab);

  Real logT0_dat[128] = {
      0.8952646494799871,0.9568554467999816,
      1.0184462441199762,1.0800370414399707,
      1.1416278387599652,1.2032186360799597,
      1.2648094333999542,1.3264002307199487,
      1.3879910280399432,1.4495818253599377,
      1.5111726226799322,1.5727634199999267,
      1.6343542173199213,1.6959450146399158,
      1.7575358119599103,1.8191266092799048,
      1.8807174065998993,1.942308203919894,
      2.0038990012398883,2.065489798559883,
      2.1270805958798773,2.188671393199872,
      2.2502621905198663,2.311852987839861,
      2.3734437851598553,2.43503458247985,
      2.4966253797998443,2.558216177119839,
      2.6198069744398333,2.681397771759828,
      2.7429885690798224,2.804579366399817,
      2.8661701637198114,2.927760961039806,
      2.989351758359801,3.050942555679795,
      3.11253335299979,3.174124150319784,
      3.235714947639779,3.297305744959773,
      3.358896542279768,3.420487339599762,
      3.482078136919757,3.543668934239751,
      3.605259731559746,3.66685052887974,
      3.728441326199735,3.790032123519729,
      3.851622920839724,3.913213718159718,
      3.974804515479713,4.036395312799707,
      4.0979861101197015,4.159576907439696,
      4.221167704759691,4.282758502079685,
      4.3443492993996795,4.405940096719674,
      4.467530894039669,4.529121691359663,
      4.5907124886796575,4.652303285999652,
      4.713894083319647,4.775484880639641,
      4.8370756779596356,4.89866647527963,
      4.960257272599625,5.02184806991962,
      5.0834388672396145,5.145029664559608,
      5.206620461879603,5.268211259199598,
      5.3298020565195925,5.391392853839586,
      5.452983651159581,5.514574448479576,
      5.5761652457995705,5.637756043119564,
      5.699346840439559,5.760937637759554,
      5.8225284350795485,5.884119232399542,
      5.945710029719537,6.007300827039532,
      6.068891624359527,6.13048242167952,
      6.192073218999515,6.25366401631951,
      6.315254813639505,6.376845610959498,
      6.438436408279493,6.500027205599488,
      6.561618002919483,6.6232088002394764,
      6.684799597559471,6.746390394879466,
      6.807981192199461,6.869571989519455,
      6.931162786839449,6.992753584159444,
      7.054344381479439,7.115935178799433,
      7.177525976119427,7.239116773439422,
      7.300707570759417,7.362298368079411,
      7.423889165399405,7.4854799627194,
      7.547070760039395,7.608661557359389,
      7.670252354679383,7.731843151999378,
      7.793433949319373,7.8550247466393675,
      7.916615543959361,7.978206341279356,
      8.03979713859935,8.101387935919345,
      8.162978733239338,8.224569530559334,
      8.286160327879328,8.347751125199323,
      8.409341922519317,8.470932719839311,
      8.532523517159307,8.5941143144793,
      8.655705111799294,8.71729590911929};

  Real mol_dat[128] = {
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2727272727272725,1.2727272727272725,
      1.2723815499493694,1.2648100812769785,
      1.2341585140707496,1.163207049051609,
      1.0864463452497828,0.999971333532216,
      0.9134963218146491,0.8403359943921913,
      0.7746439431922867,0.7214830583816662,
      0.688205616371303,0.673366301864683,
      0.6628564601867772,0.6524711027283461,
      0.64394722838337,0.6391798385193733,
      0.637230854370735,0.6363366633361379,
      0.6349034116112444,0.6309231994071257,
      0.6240559688851451,0.617126518488689,
      0.612559137915827,0.6102247390811194,
      0.6092354689367231,0.6088233578079478,
      0.6085063901426735,0.6084311002399019,
      0.6083425128382627,0.6080522194655036,
      0.6079120521788071,0.6076645705984074,
      0.6076389030859453,0.607638872252996,
      0.607638872252996,0.6076105474564117,
      0.6074081135672703,0.6073752545237256,
      0.6073752545237256,0.6073306362733513,
      0.6071862395331388,0.6071118654308184,
      0.6070987887725098,0.6069735317190199,
      0.6068487046769577,0.6067538641367652,
      0.6066225662961198,0.6065857719653424,
      0.6065857719653424,0.6065857719653424,
      0.6065857719653424,0.6065857719653424,
      0.6065857719653424,0.6063535422157312,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854,
      0.6063230669996854,0.6063230669996854};

  Real logLam_dat[128] ={
      -30.755146547390883,-30.197325878296652,
      -29.708931445722897,-29.28082869100457,
      -28.905083544768573,-28.574804822862912,
      -28.284007292438403,-28.027492699780904,
      -27.800746406060043,-27.599847585348968,
      -27.421391207141095,-27.26242025841363,
      -27.120366862646044,-26.99300112907805,
      -26.878386718348303,-26.77484224350311,
      -26.680907740820146,-26.595315545231383,
      -26.516964992328838,-26.444900444716463,
      -26.378292206324886,-26.316419945532832,
      -26.258658297667917,-26.204464360672674,
      -26.15344218991379,-26.10511496351854,
      -26.05909754672224,-26.015085995859256,
      -25.97281631694615,-25.932059220399605,
      -25.89261556375462,-25.854312392226163,
      -25.816999498761156,-25.780546435487445,
      -25.744839917383935,-25.709781566746415,
      -25.675285953757328,-25.641278894229586,
      -25.607695968084663,-25.574481166653758,
      -25.54158446468144,-25.50894053844439,
      -25.476216802704965,-25.440359284390816,
      -25.37769053824818,-25.175927078602214,
      -24.68317314983293,-24.021472574459896,
      -23.3703546464631,-22.799019763425868,
      -22.353076893424348,-22.07352829056459,
      -21.825700327545988,-21.66195201333719,
      -21.49820369912839,-21.55491004901766,
      -21.735270575719813,-21.847776287069216,
      -21.852615457421923,-21.859202399778418,
      -21.817031606331,-21.727081460251686,
      -21.61562320271156,-21.49108337973802,
      -21.360489351901855,-21.235293263209538,
      -21.118319979368238,-21.01624700781205,
      -20.937082636825064,-20.885446022240306,
      -20.87593184416351,-20.895844369396958,
      -20.899231727724732,-20.873538579486688,
      -20.85325678965284,-20.841990677001853,
      -20.850055117199226,-20.93904107454132,
      -21.158852859756358,-21.40819480495683,
      -21.544772096053954,-21.58335369236047,
      -21.592802806964652,-21.636298451817535,
      -21.739318967302843,-21.815130676644188,
      -21.83411270315918,-21.838305994759242,
      -21.83028465345719,-21.821518423892194,
      -21.83759052964146,-21.920542200599613,
      -22.061378574530774,-22.192865881496836,
      -22.283520012365177,-22.336401649145937,
      -22.374524046907027,-22.411478525299025,
      -22.444135691604394,-22.470258321388677,
      -22.479213218805615,-22.46134562748461,
      -22.45065342056148,-22.4545803148149,
      -22.48015080579247,-22.525430776514874,
      -22.562637302512044,-22.586739763269478,
      -22.60492526067487,-22.61,
      -22.60751202283908,-22.600220551218346,
      -22.59376343532172,-22.57676150624696,
      -22.55728342550207,-22.539643602599014,
      -22.520181329049628,-22.497441535712383,
      -22.478592770848717,-22.457090551423985,
      -22.43245423249599,-22.407817913567985,
      -22.38318159463999,-22.35854527571199,
      -22.333908956783993,-22.309272637855997,
      -22.284636318927998,-22.284636318927998};

  for (int n = 0; n < ntab; n++) {
    logT0(n) = logT0_dat[n];
    mol(n) = mol_dat[n];
    logLam(n) = logLam_dat[n];
  }

  logT0_min = 1.0 - std::log10(mol(0));
  logT0_max = 8.5 - std::log10(mol(ntab-1));
  dlogT0 = (logT0_max - logT0_min)/(Real) (ntab-1);
}
