/*  bentGeometry.h — THE geometry of the bent-pipe G4 case, shared.
 *
 *  One definition of the channel constants, the bend frame, the domain
 *  layout, the analytic bent Poiseuille field, and the true bore surface —
 *  included by BOTH the scalar app (bentPipe3d.cpp) and the fluid app
 *  (bentFlow3d.cpp), so the two lattices can never disagree about where
 *  the pipe is. Factored out 2026-08-28 for the solved-flow rung; the
 *  definitions are verbatim from bentPipe3d.cpp (regression-checked
 *  against a pre-refactor deterministic run).
 */

#ifndef BETAFLOW_BENT_GEOMETRY_H
#define BETAFLOW_BENT_GEOMETRY_H

#include <cmath>
#include <memory>

namespace bent {

using T = double;

// Hofmann et al. 2024 Table 1 + betaFlow's dimensional split of Pe = 200
// (betaflow/cases/mc_channel.yaml). Fixed on purpose: the runner varies
// numerics only, never physics.
static constexpr T RADIUS      = 200e-6;
static constexpr T U_MEAN      = 1.5e-3;
static constexpr T DIFFUSIVITY = 1.5e-9;
static constexpr T NU_WATER    = 1e-6;     // coupled scenario: water
static constexpr T CX          = 100e-6;
static constexpr T DBAR[3]     = {150e-6, 750e-6, 1550e-6};

// The bend frame: mother along +x to B0 = (xb, 0, 0); arc of radius R
// about C = (xb, R, 0) turning by `angle` toward +y; daughter straight
// from B1 along d2. Centreline point at arc angle phi:
//   B(phi) = C + R (sin phi, -cos phi, 0), tangent t(phi) = (cos phi,
//   sin phi, 0).
struct BendFrame {
  T xb, R, angle;
  olb::Vector<T,3> C, d2, B1;
  BendFrame(T xb_, T R_, T angle_) : xb(xb_), R(R_), angle(angle_) {
    C  = olb::Vector<T,3>(xb, R, T(0));
    d2 = olb::Vector<T,3>(std::cos(angle), std::sin(angle), T(0));
    B1 = C + olb::Vector<T,3>(R * std::sin(angle),
                              -R * std::cos(angle), T(0));
  }
  // arc angle of the point's azimuth about C (0 at bend entry)
  T phiOf(const T in[]) const {
    return std::atan2(in[0] - C[0], -(in[1] - C[1]));
  }
  // squared distance from the region-wise centreline — the ONE geometry
  // definition shared by the velocity fields and the NT solid fraction.
  T radial2(const T in[]) const {
    const T phi = (angle > T(0)) ? phiOf(in) : T(-1);
    if (angle <= T(0) || phi <= T(0) || in[0] <= xb) {
      return in[1]*in[1] + in[2]*in[2];
    } else if (phi < angle) {
      const T wx = in[0] - C[0], wy = in[1] - C[1];
      const T dR = std::sqrt(wx*wx + wy*wy) - R;
      return dR*dR + in[2]*in[2];
    }
    const olb::Vector<T,3> q(in[0] - B1[0], in[1] - B1[1], in[2]);
    const T sd = q * d2;
    const olb::Vector<T,3> rad = q - d2 * sd;
    return rad * rad;
  }
  // path coordinate along the centreline (from the mother start).
  T pathOf(const T in[]) const {
    const T phi = (angle > T(0)) ? phiOf(in) : T(-1);
    if (angle <= T(0) || phi <= T(0) || in[0] <= xb) {
      return in[0];
    } else if (phi < angle) {
      return xb + R * phi;
    }
    const olb::Vector<T,3> q(in[0] - B1[0], in[1] - B1[1], in[2]);
    return xb + R * angle + q * d2;
  }
  // local tangent direction (unit) at the point's region.
  void tangent(const T in[], T out[3]) const {
    const T phi = (angle > T(0)) ? phiOf(in) : T(-1);
    if (angle <= T(0) || phi <= T(0) || in[0] <= xb) {
      out[0] = T(1); out[1] = T(0); out[2] = T(0);
    } else if (phi < angle) {
      out[0] = std::cos(phi); out[1] = std::sin(phi); out[2] = T(0);
    } else {
      out[0] = d2[0]; out[1] = d2[1]; out[2] = T(0);
    }
  }
  // the analytic bent Poiseuille field, PHYSICAL units.
  void analyticU(const T in[], T out[3]) const {
    T t[3];
    tangent(in, t);
    const T r2 = radial2(in);
    const T u = 2.0 * U_MEAN
                * std::max(T(0), T(1) - r2 / (RADIUS*RADIUS));
    out[0] = u * t[0]; out[1] = u * t[1]; out[2] = u * t[2];
  }
};

// The domain layout both apps must agree on, computed once from the CLI
// numerics. Everything downstream (box, windows, caps, field dump grid)
// derives from these.
struct BentLayout {
  T dx, slugW, x0, tMax, drift, spread, pathLen, arcLen, daughterLen;
  T pad, xMax, yMin, yMax;
  BendFrame f;
  olb::Vector<T,3> pEnd;
  BentLayout(int res, T horizon, T angleDeg, T rBend, T sBend)
    : f(T(0), rBend, angleDeg * M_PI / 180.0) {
    dx = RADIUS / T(res);
    slugW = 3.0 * dx;                 // realised 3-slice slug extent
    x0 = 10.0 * dx;
    const T t2max = (DBAR[2] + CX/2.0) / (2.0 * U_MEAN);
    tMax = horizon * t2max;
    drift = 2.0 * U_MEAN * tMax;
    spread = 4.0 * std::sqrt(2.0 * DIFFUSIVITY * tMax);
    pathLen = std::max(drift + spread, DBAR[2] + CX) + 10.0 * dx;
    f = BendFrame(x0 + sBend, rBend, angleDeg * M_PI / 180.0);
    arcLen = rBend * f.angle;
    daughterLen = pathLen - sBend - arcLen;
    pEnd = f.B1 + f.d2 * daughterLen;
    pad = 2.0 * dx;
    xMax = std::max(f.B1[0], pEnd[0]) + RADIUS + pad;
    yMin = -(RADIUS + pad);
    yMax = std::max(RADIUS, pEnd[1] + RADIUS) + pad;
  }
};

// The true bore surface as a composite indicator (mother + 8 arc
// segments + daughter), for Bouzidi walls and any distance queries.
// Extended half a cell past both ends so cap links get distances.
inline std::shared_ptr<olb::IndicatorF3D<T>>
boreSurface(const BentLayout& L, T overshoot = T(-1)) {
  // overshoot: how far the lateral surface extends past each end. The
  // scalar app keeps the half-cell default (its ends are capped walls);
  // the FLUID app passes several cells so the surface's end caps sit
  // outside its open inlet/outlet boundary cells.
  using olb::IndicatorCylinder3D;
  using olb::IndicatorF3D;
  using olb::Vector;
  const BendFrame& f = L.f;
  const T ov = (overshoot < T(0)) ? 0.5*L.dx : overshoot;
  std::shared_ptr<IndicatorF3D<T>> surf(
    new IndicatorCylinder3D<T>(
      Vector<T,3>(2.0*L.dx - ov, T(0), T(0)),
      Vector<T,3>(f.xb, T(0), T(0)), RADIUS));
  if (f.angle > T(0)) {
    constexpr int NSEG = 8;
    for (int k = 0; k < NSEG; ++k) {
      const T p0 = f.angle * T(k) / NSEG, p1 = f.angle * T(k + 1) / NSEG;
      const Vector<T,3> a0 = f.C
        + Vector<T,3>(f.R*std::sin(p0), -f.R*std::cos(p0), T(0));
      const Vector<T,3> a1 = f.C
        + Vector<T,3>(f.R*std::sin(p1), -f.R*std::cos(p1), T(0));
      surf = surf + std::shared_ptr<IndicatorF3D<T>>(
        new IndicatorCylinder3D<T>(a0, a1, RADIUS));
    }
    surf = surf + std::shared_ptr<IndicatorF3D<T>>(
      new IndicatorCylinder3D<T>(f.B1, L.pEnd + f.d2 * ov, RADIUS));
  } else {
    surf = surf + std::shared_ptr<IndicatorF3D<T>>(
      new IndicatorCylinder3D<T>(Vector<T,3>(f.xb, T(0), T(0)),
                                 L.pEnd + f.d2 * ov, RADIUS));
  }
  return surf;
}

}  // namespace bent

#endif
