//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rt.cpp
//! \brief Problem generator for hydrostatic, gravitationally stratified atmosphere.
//! REFERENCE:
//========================================================================================

// C headers

// C++ headers
#include <cmath>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#if MAGNETIC_FIELDS_ENABLED
#error "This problem generator assumes no magnetic fields"
#endif

void ProjectPressureInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void ProjectPressureOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void MyDualEnSwitchingFunc(Real &ei_nonc,
          const Real &ei_cons, const Real &ekin, const Real &emag);

namespace {
// made global to share with BC functions
Real grav_acc, den0, tem0;
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Enroll special BCs
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, ProjectPressureInnerX2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, ProjectPressureOuterX2);
  if (DUAL_ENERGY_ENABLED)
    EnrollUserDualEnergySwitchingFunction(MyDualEnSwitchingFunc);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for gravitationally stratified atmosphere
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::int64_t iseed = -1;
  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;
  den0 = 1.0;
  tem0 = 1.0;
  // x2 is the vertical direction
  grav_acc = phydro->hsrc.GetG2();
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; i++) {
        Real den = den0 * std::exp(grav_acc * y / tem0);
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = den*tem0/gm1;
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureInnerX2(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

void ProjectPressureInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        if (n==(IVY)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IVY,k,jl-j,i) = -prim(IVY,k,jl+j-1,i);  // reflect 2-velocity
          }
        } else if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,k,jl-j,i) = prim(IPR,k,jl+j-1,i)
                                 - prim(IDN,k,jl+j-1,i)*grav_acc*(2*j-1)*pco->dx2f(j);
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,jl-j,i) = prim(n,k,jl+j-1,i);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureOuterX2(MeshBlock *pmb, Coordinates *pco,
//!                             AthenaArray<Real> &prim, FaceField &b, Real time, Real dt,
//!                             int il, int iu, int jl, int ju, int kl, int ku, int ngh)
//! \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

void ProjectPressureOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        if (n==(IVY)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IVY,k,ju+j,i) = -prim(IVY,k,ju-j+1,i);  // reflect 2-velocity
          }
        } else if (n==(IPR)) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(IPR,k,ju+j,i) = prim(IPR,k,ju-j+1,i)
                                 + prim(IDN,k,ju-j+1,i)*grav_acc*(2*j-1)*pco->dx2f(j);
          }
        } else {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            prim(n,k,ju+j,i) = prim(n,k,ju-j+1,i);
          }
        }
      }
    }
  }
  return;
}

void MyDualEnSwitchingFunc(Real &ei_nonc,
          const Real &ei_cons, const Real &ekin, const Real &emag) {
  // Synchronize ein with etot using switching function
  //ei_nonc = (ei_cons > 0.98 * ei_nonc) ? ei_cons : ei_nonc;

  // this introduces a large numerical noise
  ei_nonc = (ei_cons > ei_nonc) ? ei_cons : ei_nonc;
  return;
}
