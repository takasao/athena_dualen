//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file test_smr_eint.cpp
//! \brief Problem generator

// C headers

// C++ headers
#include <cmath>
#include <ctime>
#include <sstream>
#include <stdexcept>

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

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

namespace {

}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real bx  = pin->GetReal("problem","bx");

  Real dL  = pin->GetReal("problem","dL");
  Real pL  = pin->GetReal("problem","pL");
  Real byL = pin->GetReal("problem","byL");

  Real dR  = pin->GetReal("problem","dR");
  Real pR  = pin->GetReal("problem","pR");
  Real byR = pin->GetReal("problem","byR");

  Real gm=peos->GetGamma();

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x = pcoord->x1v(i);
        if (x < 0.0) {
          phydro->u(IDN,k,j,i) = dL;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(IEN,k,j,i) = pL / (gm-1.0);
          if (DUAL_ENERGY_ENABLED)
            phydro->u(IEI,k,j,i) = pL / (gm-1.0);
        } else {
          phydro->u(IDN,k,j,i) = dR;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(IEN,k,j,i) = pR / (gm-1.0);
          if (DUAL_ENERGY_ENABLED)
            phydro->u(IEI,k,j,i) = pR / (gm-1.0);
        }
      }
    }
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x1f(k,j,i) = bx;
        Real x = pcoord->x1f(i);
        if (x < 0.0) {
          pfield->b.x2f(k,j,i) = byL;
          pfield->b.x3f(k,j,i) = 0.0;
        } else {
          pfield->b.x2f(k,j,i) = byR;
          pfield->b.x3f(k,j,i) = 0.0;
        }

        phydro->u(IEN,k,j,i) += 0.5*(SQR(pfield->b.x1f(k,j,i))
                                   + SQR(pfield->b.x2f(k,j,i))
                                   + SQR(pfield->b.x3f(k,j,i)));
      }
    }
  }

  // end by adding bi.x1 at ie+1, bi.x2 at je+1, and bi.x3 at ke+1
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pfield->b.x1f(k,j,ie+1) = pfield->b.x1f(k,j,ie);
    }
  }

  for (int k=ks; k<=ke; ++k) {
    for (int i=is; i<=ie; ++i) {
      pfield->b.x2f(k,je+1,i) = pfield->b.x2f(k,je,i);
    }
  }
  for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
      pfield->b.x3f(ke+1,j,i) = pfield->b.x3f(ke,j,i);
    }
  }
  return;
}
