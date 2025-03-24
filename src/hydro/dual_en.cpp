//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dual_en.cpp
//! \brief functions for dual energy formalism

// C headers

// C++ headers
#include <cfloat>  // FLT_MIN

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "hydro.hpp"

namespace {
  const Real C0  = 3.0;
  const Real Cub = 1.0;
  const Real Clb = 0.98;
}

void DefaultDualEnergySwitchingFunc(Real &ei_nonc,
          const Real &ei_cons, const Real &ekin, const Real &emag);

//----------------------------------------------------------------------------------------
//! \fn void Hydro::InitializeDualEnergyFormalism
//! \brief calculate internal energy from total energy of initial condition
void Hydro::InitializeDualEnergyFormalism() {
  MeshBlock *pmb = pmy_block;
  Field *pfield = pmb->pfield;
  Coordinates *pco = pmb->pcoord;

  if (MAGNETIC_FIELDS_ENABLED) {
    // calculate bcc inside the numerical domain except for ghost cells
    pmb->pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pco,
                                            pmb->is, pmb->ie,
                                            pmb->js, pmb->je,
                                            pmb->ks, pmb->ke);
  }

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real& u_d  = u(IDN,k,j,i);
        Real& u_m1 = u(IM1,k,j,i);
        Real& u_m2 = u(IM2,k,j,i);
        Real& u_m3 = u(IM3,k,j,i);
        Real& u_e  = u(IEN,k,j,i);

        if (MAGNETIC_FIELDS_ENABLED) {
          // magnetic fields at current time
          Real& bcc1 = pfield->bcc(IB1,k,j,i);
          Real& bcc2 = pfield->bcc(IB2,k,j,i);
          Real& bcc3 = pfield->bcc(IB3,k,j,i);
          Real pb  = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
          Real e_k = 0.5*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3)) / u_d;
          // Calculate the internal energy.
          u(IEI,k,j,i) = u_e - e_k - pb;
          w(IEI,k,j,i) = u_e - e_k - pb;
        } else {
          Real e_k = 0.5*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3)) / u_d;
          // Calculate the internal energy.
          u(IEI,k,j,i) = u_e - e_k;
          w(IEI,k,j,i) = u_e - e_k;
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AnalyticIntegrationDualEnergyFormalism
//! \brief integrate the source term for internal energy equation analytically
void Hydro::AnalyticIntegrationDualEnergyFormalism(AthenaArray<Real> &u,
                      int il, int iu, int jl, int ju, int kl, int ku, const Real wght) {
  MeshBlock *pmb=pmy_block;
  Coordinates *pco = pmb->pcoord;
  Hydro *ph=pmb->phydro;
  EquationOfState *peos=pmb->peos;

  int is = il; int js = jl; int ks = kl;
  int ie = iu; int je = ju; int ke = ku;

  Real gm1 = peos->GetGamma() - 1.0;
  AthenaArray<Real> &x1area = ph->x1face_area_,
    &x2area = ph->x2face_area_, &x2area_p1 = ph->x2face_area_p1_,
    &x3area = ph->x3face_area_, &x3area_p1 = ph->x3face_area_p1_,
    &vol = ph->cell_volume_;

  AthenaArray<Real> &vf1 = ph->vf[X1DIR];
  AthenaArray<Real> &vf2 = ph->vf[X2DIR];
  AthenaArray<Real> &vf3 = ph->vf[X3DIR];
  AthenaArray<Real> &divV = ph->divV_;

  // Solve nonconservative internal energy equation
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      // calculate x1-flux divergence
      pmb->pcoord->Face1Area(k,j,is,ie+1,x1area);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        divV(i) = (x1area(i+1)*vf1(k,j,i+1)-x1area(i)*vf1(k,j,i));
      }
      // calculate x2-flux divergence
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face2Area(k,j  ,is,ie,x2area   );
        pmb->pcoord->Face2Area(k,j+1,is,ie,x2area_p1);
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          divV(i) += (x2area_p1(i)*vf2(k,j+1,i)-x2area(i)*vf2(k,j,i));
        }
      }
      // calculate x3-flux divergence
      if (pmb->block_size.nx3 > 1) {
        pmb->pcoord->Face3Area(k  ,j,is,ie,x3area   );
        pmb->pcoord->Face3Area(k+1,j,is,ie,x3area_p1);
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          divV(i) += (x3area_p1(i)*vf3(k+1,j,i)-x3area(i)*vf3(k,j,i));
        }
      }
      // update eint using non conservative form
      pmb->pcoord->CellVolume(k,j,is,ie,vol);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        // u(IEI) at this moment is the solution of the advection step only.
        Real &ei   = u(IEI,k,j,i);
        Real divV0 = divV(i)/vol(i);
        // source step (note: weighted dt must be used)
        ei = ei * std::exp(-gm1 * divV0 * wght);
      }
    }
  }

  return;
}

// note: we must use data at current time, including magnetic fields.
void Hydro::ApplyDualEnergyFormalism(AthenaArray<Real> &cons) {
  MeshBlock *pmb = pmy_block;
  Field *pfield = pmb->pfield;
  Coordinates *pco = pmb->pcoord;

  Real gm1 = pmb->peos->GetGamma() - 1.0;
  Real pressure_floor = pmb->peos->GetPressureFloor();

  // calculate bcc from pfield-b at current time
  // note1: make sure that INT_FLD is finished
  // before this function is called.
  // note2: it is assumed that STS is used when diffusion terms
  // are included.
  if (MAGNETIC_FIELDS_ENABLED)
    pfield->CalculateCellCenteredField(pfield->b,bcc_wrk_,pco,
          pmb->is,pmb->ie,pmb->js,pmb->je,pmb->ks,pmb->ke);

  Real u_ei_cons, e_k, pb;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd private(u_ei_cons, e_k, pb)
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);
        Real& u_ei = cons(IEI,k,j,i); // from nonconservative eq

        // apply pressure floor to eint
        // calculated from the non-conservative equation
        // (but do not modify total energy)
        u_ei = (gm1 * u_ei > pressure_floor) ? u_ei : (pressure_floor/gm1);

        if (MAGNETIC_FIELDS_ENABLED) {
          // magnetic fields at current time
          Real& bcc1 = bcc_wrk_(IB1,k,j,i);
          Real& bcc2 = bcc_wrk_(IB2,k,j,i);
          Real& bcc3 = bcc_wrk_(IB3,k,j,i);

          // magnetic and kinetic energy at the current time
          pb  = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
          e_k = 0.5*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3)) / u_d;

          // calculate eint at the current time
          // using the conservative equation
          u_ei_cons = u_e - e_k - pb;
        } else {
          // kinetic energy at the current time
          e_k = 0.5*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3)) / u_d;
          pb  = 0.0;
          // calculate eint at the current time
          // using the conservative equation
          u_ei_cons = u_e - e_k;
        }

        // Switching function
        if (UserDualEnergy != nullptr) {
          // user-defined switching function
          UserDualEnergy(u_ei, u_ei_cons, e_k, pb);
        } else {
          DefaultDualEnergySwitchingFunc(u_ei, u_ei_cons, e_k, pb);
        }

        // update total energy if DUAL_ENERGY_FLAG is nonzero
        if (DUAL_ENERGY_FLAG) u_e  = u_ei + e_k + pb;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::SynchronizeEintWithEtot
//! \brief calculate internal energy from total energy
void Hydro::SynchronizeEintWithEtot(AthenaArray<Real> &prim, AthenaArray<Real> &cons,
                                  AthenaArray<Real> &bcc,
                                  int il, int iu, int jl, int ju, int kl, int ku) {
  Real igm1 = 1.0 / (pmy_block->peos->GetGamma() - 1.0);
  Real pfloor = pmy_block->peos->GetPressureFloor();
  Real ein_floor = igm1 * pfloor;

  Real ein;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd private(ein)
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        if (MAGNETIC_FIELDS_ENABLED) {
          Real& bcc1 = bcc(IB1,k,j,i);
          Real& bcc2 = bcc(IB2,k,j,i);
          Real& bcc3 = bcc(IB3,k,j,i);

          Real pb  = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
          Real e_k = 0.5*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3)) / u_d;

          ein = u_e - e_k - pb;
        } else {
          Real e_k = 0.5*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3)) / u_d;
          ein = u_e - e_k;
        }

        // apply floor
        if (ein > ein_floor) {
          cons(IEI,k,j,i) = ein;
          prim(IEI,k,j,i) = ein;
        } else {
          cons(IEI,k,j,i) = ein_floor;
          prim(IEI,k,j,i) = ein_floor;
        }
      }
    }
  }
  return;
}

void DefaultDualEnergySwitchingFunc(Real &ei_nonc,
          const Real &ei_cons, const Real &ekin, const Real &emag) {
  Real alpha = (ekin+emag) / std::max(ei_nonc,1e-18); // avoid zero-division
  Real tmp = alpha/(C0 + alpha);
  Real C = std::min(Cub, std::max(Clb, tmp));

  // Synchronize ein with etot using switching function
  ei_nonc = (ei_cons > (C*ei_nonc)) ? ei_cons : ei_nonc;
  return;
}
