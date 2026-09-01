/*  junctionGeometry.h — THE geometry of the Y-junction case, shared.
 *
 *  The pre-registered bifurcation (docs/bifurcation-preregistration.md):
 *  mother = the Table-1 pipe along +x; at the branch point two identical
 *  daughters at ±30° with Murray's radius for an equal split,
 *  a_d = a·2^(−1/3). One definition of the frame, the bore, the layout,
 *  the warm-start field and the window/section geometry — included by
 *  BOTH the fluid app (junctionFlow3d) and the scalar app
 *  (junctionPipe3d), so the lattices cannot disagree about the vessel.
 *
 *  The apex: the mother bore ends at the branch plane x = xj; each
 *  daughter bore is a cylinder from B = (xj, 0, 0) back-extended by one
 *  mother radius, so their union forms the junction throat and the
 *  carina emerges where the two daughter surfaces intersect. This is the
 *  first junction geometry of the programme, stated rather than tuned;
 *  smoother carinas are later variants.
 */

#ifndef BETAFLOW_JUNCTION_GEOMETRY_H
#define BETAFLOW_JUNCTION_GEOMETRY_H

#include <cmath>

namespace junction {

using T = double;

// The channel constants (Hofmann Table 1 + betaFlow's split), identical
// to bentGeometry.h on purpose — the junction descends from the same case.
static constexpr T RADIUS      = 200e-6;
static constexpr T U_MEAN      = 1.5e-3;
static constexpr T DIFFUSIVITY = 1.5e-9;
static constexpr T NU_WATER    = 1e-6;
static constexpr T CX          = 100e-6;
static constexpr T DBAR[3]     = {150e-6, 750e-6, 1550e-6};

// Murray's law for an equal split into two daughters: a^3 = 2 a_d^3.
static const T MURRAY   = std::pow(2.0, -1.0/3.0);   // 0.7937
static const T RADIUS_D = RADIUS * MURRAY;           // 158.7 um
static const T U_MEAN_D = U_MEAN * MURRAY;           // 1.19 mm/s (per branch)

struct JunctionFrame {
  T xj, angle;                       // branch plane; half-angle
  olb::Vector<T,3> dPlus, dMinus, B;
  JunctionFrame(T xj_, T angleDeg)
    : xj(xj_), angle(angleDeg * M_PI / 180.0) {
    dPlus  = olb::Vector<T,3>(std::cos(angle),  std::sin(angle), T(0));
    dMinus = olb::Vector<T,3>(std::cos(angle), -std::sin(angle), T(0));
    B      = olb::Vector<T,3>(xj, T(0), T(0));
  }
  // squared radial distance from the mother axis
  T r2Mother(const T in[]) const {
    return in[1]*in[1] + in[2]*in[2];
  }
  // along-axis coordinate and squared radial distance for daughter s=±1
  void daughterCoords(const T in[], int sgn, T& s, T& r2) const {
    const olb::Vector<T,3>& d = (sgn > 0) ? dPlus : dMinus;
    const olb::Vector<T,3> q(in[0] - B[0], in[1] - B[1], in[2]);
    s = q * d;
    const olb::Vector<T,3> rad = q - d * s;
    r2 = rad * rad;
  }
  // inside the bore? (mother capped at the branch plane; daughters
  // back-extended one mother radius to form the throat)
  bool insideBore(const T in[], T daughterLen) const {
    if (in[0] <= xj && r2Mother(in) <= RADIUS*RADIUS) { return true; }
    for (int sgn : {+1, -1}) {
      T s, r2;
      daughterCoords(in, sgn, s, r2);
      if (s >= -RADIUS && s <= daughterLen && r2 <= RADIUS_D*RADIUS_D) {
        return true;
      }
    }
    return false;
  }
  // warm-start / reference field, PHYSICAL units: mother Poiseuille up
  // to the branch, Murray-scaled Poiseuille along each daughter beyond;
  // in the overlap the larger magnitude wins (warm start only — the
  // solved field replaces all of it).
  void warmU(const T in[], T out[3], T daughterLen) const {
    out[0] = out[1] = out[2] = T(0);
    T best = T(0);
    if (in[0] <= xj + RADIUS) {
      const T u = 2.0 * U_MEAN
        * std::max(T(0), T(1) - r2Mother(in)/(RADIUS*RADIUS));
      if (u > best) { best = u; out[0] = u; out[1] = T(0); out[2] = T(0); }
    }
    for (int sgn : {+1, -1}) {
      T s, r2;
      daughterCoords(in, sgn, s, r2);
      if (s >= -RADIUS && s <= daughterLen + RADIUS) {
        const T u = 2.0 * U_MEAN_D
          * std::max(T(0), T(1) - r2/(RADIUS_D*RADIUS_D));
        if (u > best) {
          best = u;
          const olb::Vector<T,3>& d = (sgn > 0) ? dPlus : dMinus;
          out[0] = u*d[0]; out[1] = u*d[1]; out[2] = u*d[2];
        }
      }
    }
  }
};

// The layout: slug in the mother at x0; branch at path sJunction from
// the slug centre; receiver path distances DBAR measured along
// mother-then-daughter(+) centreline, with mirror windows in daughter(−).
struct JunctionLayout {
  T dx, slugW, x0, tMax, pathLen, daughterLen;
  T pad, xMax, yAbsMax;
  JunctionFrame f;
  olb::Vector<T,3> pEndPlus, pEndMinus;
  JunctionLayout(int res, T horizon, T angleDeg, T sJunction)
    : f(T(0), angleDeg) {
    dx = RADIUS / T(res);
    slugW = 3.0 * dx;
    x0 = 10.0 * dx;
    const T t2max = (DBAR[2] + CX/2.0) / (2.0 * U_MEAN);
    tMax = horizon * t2max;
    // Daughter speed is lower, so drift shrinks; the window coverage
    // term still dominates the length requirement.
    const T drift = 2.0 * U_MEAN * tMax;
    const T spread = 4.0 * std::sqrt(2.0 * DIFFUSIVITY * tMax);
    pathLen = std::max(drift + spread, DBAR[2] + CX) + 10.0 * dx;
    f = JunctionFrame(x0 + sJunction, angleDeg);
    daughterLen = pathLen - sJunction;
    pEndPlus  = f.B + f.dPlus  * daughterLen;
    pEndMinus = f.B + f.dMinus * daughterLen;
    pad = 2.0 * dx;
    xMax = std::max(f.xj, pEndPlus[0]) + RADIUS + pad;
    // SNAPPED to the grid: mirror symmetry about y = 0 requires the box's
    // y-extent to be a whole number of cells, or cell centres on the two
    // sides do not mirror (measured at res 6 before the snap: outlet
    // discs of 63 vs 69 cells and a 7% flux asymmetry - gate G3 catching
    // its exact target class, a broken-alignment bug, before physics).
    yAbsMax = std::ceil((pEndPlus[1] + RADIUS_D + pad) / dx) * dx;
  }
  // window centre+axis for path distance dbar: mother frame before the
  // junction, daughter(sgn) frame after.
  void windowEnds(T dbar, int sgn, olb::Vector<T,3>& w0,
                  olb::Vector<T,3>& w1) const {
    if (dbar + CX/2.0 <= f.xj - x0) {
      w0 = olb::Vector<T,3>(x0 + dbar - CX/2.0, T(0), T(0));
      w1 = olb::Vector<T,3>(x0 + dbar + CX/2.0, T(0), T(0));
    } else {
      const olb::Vector<T,3>& d = (sgn > 0) ? f.dPlus : f.dMinus;
      const T sOut = (x0 + dbar) - f.xj;     // path past the branch
      w0 = f.B + d * (sOut - CX/2.0);
      w1 = f.B + d * (sOut + CX/2.0);
    }
  }
};

}  // namespace junction

#endif
