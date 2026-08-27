/*  bentPipe3d — gate G4 of the bifurcation pre-registration.
 *
 *  The bent-pipe control (docs/bifurcation-preregistration.md): the
 *  mc_channel scalar transport through a pipe that runs straight, turns
 *  through --angle degrees along a circular arc of radius --rbend, and
 *  continues straight. At angle 0 this is the straight case rebuilt
 *  through ALL the new junction machinery — composite geometry, a
 *  region-wise analytic velocity field, path-based receiver windows,
 *  capped ends — so the straight record is the known answer and any
 *  deviation is machinery, not physics. At angle 30 the deviation from
 *  the angle-0 control is the bend, reported per the pre-registration.
 *
 *  CONSTRUCTION HISTORY, kept per the correction policy. Version 1 used a
 *  MITRED elbow: two cylinders joined at a plane, the velocity switching
 *  direction discontinuously across it. Measured 2026-08-26: the scalar
 *  lattice DIVERGES at the mitre at the stability-pinned tau (bulk mass
 *  -1.9e51 by the horizon at tau = 0.5048; still -5.8e46 at a doubled
 *  margin tau = 0.5096, so damping was the wrong knob). Plane-wave
 *  stability of the uniform oblique scheme is clean (max |lambda| < 1,
 *  scan kept in tests/test_bifurcation_g4.py), which localises the
 *  mechanism to the kink itself: a frozen field with a grid-scale
 *  direction discontinuity feeds a local instability that relaxation
 *  cannot damp at any operable tau. This version therefore bends the
 *  CENTRELINE smoothly: direction is continuous everywhere (C0, with the
 *  arc tangent varying linearly in path), which removes the grid-scale
 *  kink by construction. The residual frozen-field artifact is the smooth
 *  divergence of U(r) t_hat(s) inside the bend, O(u a / rbend), reported
 *  with the results rather than hidden.
 *
 *  LAYOUT: the bend (arc length rbend * angle) sits BETWEEN receiver
 *  windows 1 and 2 — bend start at path --sbend (default 245 um) from the
 *  slug centre, so with rbend = 2a and 30 degrees the arc spans
 *  245..454 um. Window 1 (100..200 um) stays in the straight mother and
 *  is an IN-RUN CONTROL: at any angle it must reproduce the straight
 *  record. Windows 2 and 3 lie wholly in the outgoing straight section.
 *
 *  Geometry along the centreline: mother cylinder to the bend entry B0;
 *  the arc as a union of 8 short cylinder segments between points of the
 *  exact arc (chord sagitta at 30/8 degrees and rbend = 2a is ~0.2 um,
 *  far below dx); daughter cylinder from the bend exit B1 along
 *  d2 = (cos a, sin a, 0). At angle 0 every piece is collinear and the
 *  union is exactly one straight cylinder.
 *
 *  CLI: --resolution --tau --horizon --outputs --outdir as mcChannel3d,
 *  plus --angle (degrees, default 0), --rbend (m, default 400e-6),
 *  --sbend (m, default 245e-6).
 */

#include <olb.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

using namespace olb;
using namespace olb::names;

// Per-cell solid fraction for the Noble-Torczynski wall (path (a) of the
// oblique-wall programme, results/oblique_wall_scheme_study.json).
struct SOLID_FRACTION : public descriptors::FIELD_BASE<1> { };

using MyCase = Case<
  AdvectionDiffusion,
  Lattice<double, descriptors::D3Q7<descriptors::VELOCITY, SOLID_FRACTION>>
>;
using T = MyCase::value_t;
using DESCRIPTOR = MyCase::descriptor_t_of<AdvectionDiffusion>;

// --dynamics bgk2: the second-order ADE equilibrium (the tensor rung of the
// analytic reference characterises exactly what it can and cannot cancel
// on D3Q7). A probe in the wall-loop instability ladder.
using AdeSecondOrderBGKdynamics = dynamics::Tuple<
  T, DESCRIPTOR,
  momenta::AdvectionDiffusionBulkTuple,
  equilibria::SecondOrder,
  collision::BGK,
  AdvectionDiffusionExternalVelocityCollision
>;

static constexpr T RADIUS      = 200e-6;
static constexpr T U_MEAN      = 1.5e-3;
static constexpr T DIFFUSIVITY = 1.5e-9;
static constexpr T CX          = 100e-6;
static constexpr T DBAR[3]     = {150e-6, 750e-6, 1550e-6};
static constexpr T CS2_D3Q7    = 0.25;

// Noble-Torczynski partially-saturated ADE dynamics: collision blends BGK
// with a pairwise-antisymmetric solid operator by B(eps, tau). On D3Q7 the
// opposite-direction weights are equal, so the u_w = 0 solid operator
// reduces to Omega_s_i = g_ibar - g_i exactly (the u_w = local variant
// diverges under plug flow - measured in the reference-lattice study).
// Exactly mass-conserving: Omega_s sums to zero pairwise.
template<typename T_, typename DESCRIPTOR_,
         typename MOMENTA=momenta::AdvectionDiffusionBulkTuple>
struct NTAdeBGKdynamics final
  : public dynamics::CustomCollision<T_,DESCRIPTOR_,MOMENTA> {
  using MomentaF = typename MOMENTA::template type<DESCRIPTOR_>;
  using EquilibriumF =
    typename equilibria::FirstOrder::template type<DESCRIPTOR_,MOMENTA>;
  using parameters = meta::list<descriptors::OMEGA, collision::TRT::MAGIC>;
  template <typename NEW_T>
  using exchange_value_type = NTAdeBGKdynamics<NEW_T,DESCRIPTOR_,MOMENTA>;
  template<typename M>
  using exchange_momenta = NTAdeBGKdynamics<T_,DESCRIPTOR_,M>;
  std::type_index id() override { return typeid(NTAdeBGKdynamics); }
  AbstractParameters<T_,DESCRIPTOR_>& getParameters(
      BlockLattice<T_,DESCRIPTOR_>& block) override {
    return block.template getData<OperatorParameters<NTAdeBGKdynamics>>();
  }
  template <typename CELL, typename PARAMETERS,
            typename V=typename CELL::value_t>
  CellStatistic<V> collide(CELL& cell, PARAMETERS& parameters) any_platform {
    // The bulk part is TRT (OMEGA relaxes the EVEN sector, MAGIC places
    // the ODD rate that carries D — the convention of the plain TRT
    // path); BGK is the MAGIC = (tau-1/2)^2 member. The solid part uses
    // the ODD rate's tau in B, since that is the transport clock.
    const auto u = cell.template getField<descriptors::VELOCITY>();
    const V C = MomentaF().computeRho(cell);
    const V omE = parameters.template get<descriptors::OMEGA>();
    const V magic = parameters.template get<collision::TRT::MAGIC>();
    const V omO = V{1} / (magic / (V{1}/omE - V{0.5}) + V{0.5});
    const V tauO = V{1} / omO;
    const V eps = cell.template getField<SOLID_FRACTION>();
    const V B = eps * (tauO - V{0.5}) / ((V{1} - eps) + (tauO - V{0.5}));
    V g0[DESCRIPTOR_::q];
    for (int i = 0; i < DESCRIPTOR_::q; ++i) { g0[i] = cell[i]; }
    for (int i = 0; i < DESCRIPTOR_::q; ++i) {
      const int io = descriptors::opposite<DESCRIPTOR_>(i);
      const V fEq  = equilibrium<DESCRIPTOR_>::firstOrder(i, C, u);
      const V fEqO = equilibrium<DESCRIPTOR_>::firstOrder(io, C, u);
      const V gP = V{0.5}*(g0[i] + g0[io]), gM = V{0.5}*(g0[i] - g0[io]);
      const V eP = V{0.5}*(fEq + fEqO),    eM = V{0.5}*(fEq - fEqO);
      cell[i] = g0[i]
              + (V{1} - B) * (-omE*(gP - eP) - omO*(gM - eM))
              + B * (g0[io] - g0[i]);
    }
    return {C, V{0}};
  };
  void computeEquilibrium(ConstCell<T_,DESCRIPTOR_>& cell, T_ rho,
                          const T_ u[DESCRIPTOR_::d],
                          T_ fEq[DESCRIPTOR_::q]) const override {
    for (int iPop = 0; iPop < DESCRIPTOR_::q; ++iPop) {
      fEq[iPop] = equilibrium<DESCRIPTOR_>::firstOrder(iPop, rho, u);
    }
  };
  std::string getName() const override {
    return "NTAdeBGKdynamics<" + MomentaF().getName() + ">";
  };
};

// The bend frame: mother along +x to B0 = (xb, 0, 0); arc of radius R
// about C = (xb, R, 0) turning by `angle` toward +y; daughter straight
// from B1 along d2. Centreline point at arc angle phi:
//   B(phi) = C + R (sin phi, -cos phi, 0),  tangent t(phi) = (cos phi,
//   sin phi, 0).
struct BendFrame {
  T xb, R, angle;
  Vector<T,3> C, d2, B1;
  BendFrame(T xb_, T R_, T angle_) : xb(xb_), R(R_), angle(angle_) {
    C  = Vector<T,3>(xb, R, T(0));
    d2 = Vector<T,3>(std::cos(angle), std::sin(angle), T(0));
    B1 = C + Vector<T,3>(R * std::sin(angle), -R * std::cos(angle), T(0));
  }
  // arc angle of the point's azimuth about C (0 at bend entry)
  T phiOf(const T in[]) const {
    return std::atan2(in[0] - C[0], -(in[1] - C[1]));
  }
  // squared distance from the region-wise centreline — the ONE geometry
  // definition shared by the velocity field and the NT solid fraction.
  T radial2(const T in[]) const {
    const T phi = (angle > T(0)) ? phiOf(in) : T(-1);
    if (angle <= T(0) || phi <= T(0) || in[0] <= xb) {
      return in[1]*in[1] + in[2]*in[2];
    } else if (phi < angle) {
      const T wx = in[0] - C[0], wy = in[1] - C[1];
      const T dR = std::sqrt(wx*wx + wy*wy) - R;
      return dR*dR + in[2]*in[2];
    }
    const Vector<T,3> q(in[0] - B1[0], in[1] - B1[1], in[2]);
    const T sd = q * d2;
    const Vector<T,3> rad = q - d2 * sd;
    return rad * rad;
  }
  // path coordinate along the centreline (from the mother start), for the
  // axial caps of the NT solid fraction.
  T pathOf(const T in[]) const {
    const T phi = (angle > T(0)) ? phiOf(in) : T(-1);
    if (angle <= T(0) || phi <= T(0) || in[0] <= xb) {
      return in[0];
    } else if (phi < angle) {
      return xb + R * phi;
    }
    const Vector<T,3> q(in[0] - B1[0], in[1] - B1[1], in[2]);
    return xb + R * angle + q * d2;
  }
};

// Region-wise analytic Poiseuille in LATTICE units, direction continuous
// everywhere. Angle 0 reduces exactly to the straight analytic field.
class BentPoiseuilleVelocity : public AnalyticalF3D<T,T> {
  T _convVel;
  BendFrame _f;
public:
  BentPoiseuilleVelocity(T convVel, BendFrame f)
    : AnalyticalF3D<T,T>(3), _convVel(convVel), _f(f) {}
  bool operator()(T out[], const T in[]) override {
    T tx, ty;
    const T phi = (_f.angle > T(0)) ? _f.phiOf(in) : T(-1);
    if (_f.angle <= T(0) || phi <= T(0) || in[0] <= _f.xb) {
      tx = T(1); ty = T(0);                       // mother frame
    } else if (phi < _f.angle) {
      tx = std::cos(phi); ty = std::sin(phi);     // bend tangent
    } else {
      tx = _f.d2[0]; ty = _f.d2[1];               // daughter frame
    }
    const T r2 = _f.radial2(in);
    const T u = 2.0 * U_MEAN
                * util::max(T(0), T(1) - r2 / (RADIUS*RADIUS)) / _convVel;
    out[0] = u * tx;
    out[1] = u * ty;
    out[2] = T(0);
    return true;
  }
};

// Solid fraction for the Noble-Torczynski wall: 4^3 supersampling of the
// true surface (the same radial2/path definitions as the velocity field,
// so wall and flow can never disagree on where the bore is). Axial caps at
// the mother start and daughter end count as solid.
class SolidFraction : public AnalyticalF3D<T,T> {
  BendFrame _f;
  T _dx, _pathMin, _pathMax;
public:
  SolidFraction(BendFrame f, T dx, T pathMin, T pathMax)
    : AnalyticalF3D<T,T>(1), _f(f), _dx(dx),
      _pathMin(pathMin), _pathMax(pathMax) {}
  bool operator()(T out[], const T in[]) override {
    int solid = 0;
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          const T off[3] = {(T(a) - 1.5) / 4.0 * _dx,
                            (T(b) - 1.5) / 4.0 * _dx,
                            (T(c) - 1.5) / 4.0 * _dx};
          const T pp[3] = {in[0] + off[0], in[1] + off[1], in[2] + off[2]};
          const T sPath = _f.pathOf(pp);
          if (_f.radial2(pp) > RADIUS*RADIUS
              || sPath < _pathMin || sPath > _pathMax) {
            ++solid;
          }
        }
      }
    }
    out[0] = T(solid) / T(64);
    return true;
  }
};

// Uniform slug in the straight mother, radial guard as in mcChannel3d.
class SlugInit : public AnalyticalF3D<T,T> {
  T _x0, _w;
public:
  SlugInit(T x0, T w) : AnalyticalF3D<T,T>(1), _x0(x0), _w(w) {}
  bool operator()(T out[], const T in[]) override {
    const T r2 = in[1]*in[1] + in[2]*in[2];
    out[0] = (std::fabs(in[0] - _x0) <= _w / 2.0
              && r2 <= RADIUS*RADIUS) ? T(1) : T(0);
    return true;
  }
};

static T argOpt(int argc, char** argv, const char* name, T fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) { return std::atof(argv[i+1]); }
  }
  return fallback;
}

static std::string argStr(int argc, char** argv, const char* name,
                          const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) { return argv[i+1]; }
  }
  return fallback;
}

int main(int argc, char* argv[]) {
  OstreamManager clout(std::cout, "bentPipe3d");
  initialize(&argc, &argv);

  const int  res     = int(argOpt(argc, argv, "--resolution", 15));
  const T    tau     = argOpt(argc, argv, "--tau", 0.6);
  const T    horizon = argOpt(argc, argv, "--horizon", 6.5);
  const int  outputs = int(argOpt(argc, argv, "--outputs", 160));
  const T    angleDeg = argOpt(argc, argv, "--angle", 0.0);
  const T    rBend   = argOpt(argc, argv, "--rbend", 400e-6);
  const T    sBend   = argOpt(argc, argv, "--sbend", 245e-6);
  // --dynamics trt: TRT collision with the ODD rate carrying the physical
  // diffusivity (measured on the exact dispersion relation: D rides the
  // odd rate in OpenLB's convention) and the EVEN rate free at --taueven.
  // The BGK wall-loop instability under oblique advection (see header) is
  // the reason this switch exists.
  const std::string dyn = argStr(argc, argv, "--dynamics", "bgk");
  const T    tauEven = argOpt(argc, argv, "--taueven", 1.0);
  // --wall bouzidi: interpolated no-flux reflection at the TRUE surface.
  // Plain (zero-wall-velocity) Bouzidi applied to the scalar populations
  // IS a no-flux wall: the reflection point sits where the prescribed
  // velocity actually vanishes, so oblique staircase faces stop receiving
  // the advective flux that feeds the bounce-back wall loop (the G4
  // instability finding). The fluid rung measured this scheme's wall
  // placement: shift ~ dx^2, order 2.1.
  const std::string wall = argStr(argc, argv, "--wall", "bb");
  const std::string outdir = argStr(argc, argv, "--outdir", "./tmp/");
  singleton::directories().setOutputDir(outdir);

  const T dx = RADIUS / T(res);
  const T dt = (tau - 0.5) * CS2_D3Q7 * dx * dx / DIFFUSIVITY;

  const T t2max = (DBAR[2] + CX/2.0) / (2.0 * U_MEAN);
  const T tMax  = horizon * t2max;
  const int iTmax = int(std::ceil(tMax / dt));
  const int statIter = util::max(1, iTmax / outputs);

  // Slug half-width 1.01 dx: the straight app's |x - x0| <= dx puts the
  // boundary EXACTLY on cell centres, and rounding included 3 slices
  // there but 2 here (measured: initial totals 1317 vs 878, ratio 1.5 -
  // the whole machinery-gate peak discrepancy). The 1% margin makes the
  // 3-slice slug deterministic; slugW below is the REALISED extent.
  const T slugHalf = 1.01 * dx;
  const T slugW = 3.0 * dx;
  const T x0 = 10.0 * dx;
  const T drift = 2.0 * U_MEAN * tMax;
  const T spread = 4.0 * std::sqrt(2.0 * DIFFUSIVITY * tMax);
  const T pathLen = util::max(drift + spread, DBAR[2] + CX) + 10.0 * dx;

  const T angle = angleDeg * M_PI / 180.0;
  const BendFrame f(x0 + sBend, rBend, angle);
  const T arcLen = rBend * angle;
  const T daughterLen = pathLen - sBend - arcLen;
  const Vector<T,3> pEnd = f.B1 + f.d2 * daughterLen;

  const T pad = 2.0 * dx;
  const T xMax = util::max(f.B1[0], pEnd[0]) + RADIUS + pad;
  const T yMin = -(RADIUS + pad);
  const T yMax = util::max(RADIUS, pEnd[1] + RADIUS) + pad;
  Vector<T,3> extent(xMax, yMax - yMin, 2.0*(RADIUS + pad));
  Vector<T,3> origin(T(0), yMin, -(RADIUS + pad));
  IndicatorCuboid3D<T> box(extent, origin);

#ifdef PARALLEL_MODE_MPI
  const int noOfCuboids = singleton::mpi().getSize();
#else
  const int noOfCuboids = 1;
#endif
  Mesh<T,MyCase::d> mesh(box, dx, noOfCuboids);
  mesh.setOverlap(2);
  mesh.getCuboidDecomposition().setPeriodicity({false, false, false});

  MyCase::ParametersD params;
  MyCase myCase(params, mesh);

  auto& geometry = myCase.getGeometry();
  geometry.rename(0, 2);
  IndicatorCylinder3D<T> mother(Vector<T,3>(2.0*dx, T(0), T(0)),
                                Vector<T,3>(f.xb, T(0), T(0)), RADIUS);
  geometry.rename(2, 1, mother);
  if (angle > T(0)) {
    constexpr int NSEG = 8;
    for (int k = 0; k < NSEG; ++k) {
      const T p0 = angle * T(k) / NSEG, p1 = angle * T(k + 1) / NSEG;
      const Vector<T,3> a0 = f.C
        + Vector<T,3>(rBend*std::sin(p0), -rBend*std::cos(p0), T(0));
      const Vector<T,3> a1 = f.C
        + Vector<T,3>(rBend*std::sin(p1), -rBend*std::cos(p1), T(0));
      IndicatorCylinder3D<T> seg(a0, a1, RADIUS);
      geometry.rename(2, 1, seg);
    }
    IndicatorCylinder3D<T> daughter(f.B1, pEnd, RADIUS);
    geometry.rename(2, 1, daughter);
  } else {
    IndicatorCylinder3D<T> rest(Vector<T,3>(f.xb, T(0), T(0)),
                                pEnd, RADIUS);
    geometry.rename(2, 1, rest);
  }
  // Receiver windows as their own BULK materials (11, 12, 13), carved
  // out of material 1: every readout below is then a material sum, which
  // is bulk-only BY CONSTRUCTION. The geometric-cylinder windows of the
  // straight app were only accidentally bulk-only (wall cells held zero
  // under bounce-back); under the NT wall, cut and solid cells carry
  // rattling in-transit scalar that a geometric window would alias.
  for (int w = 0; w < 3; ++w) {
    Vector<T,3> w0, w1;
    if (DBAR[w] + CX/2.0 <= sBend) {
      w0 = Vector<T,3>(x0 + DBAR[w] - CX/2.0, T(0), T(0));
      w1 = Vector<T,3>(x0 + DBAR[w] + CX/2.0, T(0), T(0));
    } else {
      const T sOut = DBAR[w] - sBend - arcLen;
      w0 = f.B1 + f.d2 * (sOut - CX/2.0);
      w1 = f.B1 + f.d2 * (sOut + CX/2.0);
    }
    IndicatorCylinder3D<T> win(w0, w1, RADIUS);
    geometry.rename(1, 11 + w, win);
  }
  geometry.communicate();
  geometry.print();

  auto& lattice = myCase.getLattice(AdvectionDiffusion{});
  lattice.setUnitConverter<AdeUnitConverter<T,DESCRIPTOR>>(
    dx, dt, RADIUS, 2.0*U_MEAN, DIFFUSIVITY, T(1));
  const auto& converter = lattice.getUnitConverter();
  converter.print();

  if (dyn == "trt") {
    dynamics::set<AdvectionDiffusionTRTdynamics>(
      lattice, geometry.getMaterialIndicator({1, 11, 12, 13}));
  } else if (dyn == "bgk2") {
    dynamics::set<AdeSecondOrderBGKdynamics>(
      lattice, geometry.getMaterialIndicator({1, 11, 12, 13}));
  } else {
    dynamics::set<AdvectionDiffusionBGKdynamics>(
      lattice, geometry.getMaterialIndicator({1, 11, 12, 13}));
  }
  if (wall == "nt") {
    // Noble-Torczynski: NO link boundary at all — the wall IS the
    // solid-fraction field, and every cell (bulk, cut, deep solid) runs
    // the blended dynamics. eps = 0 recovers pure BGK in the bulk;
    // eps = 1 is a pure pairwise reflection that rattles in place.
    dynamics::set<NTAdeBGKdynamics>(
      lattice, geometry.getMaterialIndicator({1, 2, 11, 12, 13}));
    SolidFraction epsF(f, dx, 2.0*dx, f.xb + rBend*angle + daughterLen);
    fields::set<SOLID_FRACTION>(
      lattice, geometry.getMaterialIndicator({1, 2, 11, 12, 13}), epsF);
  } else if (wall == "bouzidi") {
    // The TRUE surface: the same composite the geometry staircase
    // approximates, extended half a cell at both ends so cap links get
    // distances (the pipeFlow3d pattern).
    std::shared_ptr<IndicatorF3D<T>> surf(
      new IndicatorCylinder3D<T>(
        Vector<T,3>(2.0*dx - 0.5*dx, T(0), T(0)),
        Vector<T,3>(f.xb, T(0), T(0)), RADIUS));
    if (angle > T(0)) {
      constexpr int NSEG = 8;
      for (int k = 0; k < NSEG; ++k) {
        const T p0 = angle * T(k) / NSEG, p1 = angle * T(k + 1) / NSEG;
        const Vector<T,3> a0 = f.C
          + Vector<T,3>(rBend*std::sin(p0), -rBend*std::cos(p0), T(0));
        const Vector<T,3> a1 = f.C
          + Vector<T,3>(rBend*std::sin(p1), -rBend*std::cos(p1), T(0));
        surf = surf + std::shared_ptr<IndicatorF3D<T>>(
          new IndicatorCylinder3D<T>(a0, a1, RADIUS));
      }
      surf = surf + std::shared_ptr<IndicatorF3D<T>>(
        new IndicatorCylinder3D<T>(f.B1, pEnd + f.d2 * (0.5*dx), RADIUS));
    } else {
      surf = surf + std::shared_ptr<IndicatorF3D<T>>(
        new IndicatorCylinder3D<T>(Vector<T,3>(f.xb, T(0), T(0)),
                                   pEnd + f.d2 * (0.5*dx), RADIUS));
    }
    setBouzidiBoundary<T, DESCRIPTOR, BouzidiPostProcessor>(
      lattice, geometry, 2, *surf);
  } else {
    boundary::set<boundary::BounceBack>(lattice, geometry, 2);
  }

  BentPoiseuilleVelocity uF(converter.getConversionFactorVelocity(), f);
  SlugInit slugF(x0, 2.0 * slugHalf);
  auto everything = geometry.getMaterialIndicator({1, 2, 11, 12, 13});
  fields::set<descriptors::VELOCITY>(lattice, everything, uF);
  AnalyticalConst3D<T,T> zeroRho(T(0));
  AnalyticalConst3D<T,T> zeroU(T(0), T(0), T(0));
  lattice.iniEquilibrium(geometry.getMaterialIndicator({1, 11, 12, 13}), slugF, uF);
  lattice.iniEquilibrium(geometry.getMaterialIndicator({2}), zeroRho, zeroU);
  if (dyn == "trt" || wall == "nt") {
    // OMEGA relaxes the EVEN sector (free choice); MAGIC places the ODD
    // rate, which carries D: Lambda = (tau_even - 1/2)(tau_ade - 1/2).
    // For NT with dyn=bgk the BGK-equivalent member is tau_even = tau.
    const T tE = (dyn == "trt") ? tauEven : tau;
    lattice.setParameter<descriptors::OMEGA>(T(1) / tE);
    lattice.setParameter<collision::TRT::MAGIC>(
      (tE - 0.5) * (tau - 0.5));
  } else {
    lattice.setParameter<descriptors::OMEGA>(
      converter.getLatticeAdeRelaxationFrequency());
  }
  lattice.initialize();

  clout << "betaflow-provenance"
        << " wall=" << wall
        << " dynamics=" << dyn
        << " taueven=" << (dyn == "trt" ? tauEven : tau)
        << " velocity_source=regionwise-analytic-arc"
        << " angle_deg=" << angleDeg
        << " rbend=" << rBend
        << " sbend=" << sBend
        << " arclen=" << arcLen
        << " tau_requested=" << tau
        << " omega=" << converter.getLatticeAdeRelaxationFrequency()
        << " dx=" << dx << " dt=" << dt
        << " uLatCentre=" << 2.0*U_MEAN / converter.getConversionFactorVelocity()
        << " iTmax=" << iTmax << " statIter=" << statIter
        << " pathLen=" << pathLen << " x0=" << x0 << " slugW=" << slugW
        << std::endl;

  std::ofstream csv(outdir + "cir.csv");
  csv.precision(12);
  csv << "# t_phys, cir150, cir750, cir1550, total_mass\n";

  util::Timer<T> timer(iTmax, geometry.getStatistics().getNvoxel());
  timer.start();

  for (int iT = 0; iT <= iTmax; ++iT) {
    if (iT % statIter == 0) {
      lattice.setProcessingContext(ProcessingContext::Evaluation);
      SuperLatticeDensity3D<T,DESCRIPTOR> rho(lattice);
      T total[1] = {T(0)};
      int tmp[1] = {0};
      // Bulk-only by construction: material sums over {1, 11, 12, 13}.
      total[0] = T(0);
      for (int mat : {1, 11, 12, 13}) {
        T part[1] = {T(0)};
        SuperSum3D<T,T> matSum(rho, geometry, mat);
        matSum(part, tmp);
        total[0] += part[0];
      }

      T cir[3];
      for (int w = 0; w < 3; ++w) {
        SuperSum3D<T,T> winSum(
          std::unique_ptr<SuperF3D<T,T>>(
            new SuperLatticeDensity3D<T,DESCRIPTOR>(lattice)),
          std::unique_ptr<SuperIndicatorF3D<T>>(
            new SuperIndicatorMaterial3D<T>(geometry,
                std::vector<int>{11 + w})));
        T out[1] = {T(0)};
        winSum(out, tmp);
        cir[w] = (total[0] > T(0)) ? out[0] / total[0] : T(0);
      }
      csv << converter.getPhysTime(iT) << ", " << cir[0] << ", " << cir[1]
          << ", " << cir[2] << ", " << total[0] << "\n";
      csv.flush();
      timer.update(iT);
      timer.printStep();
    }
    lattice.collideAndStream();
  }

  timer.stop();
  timer.printSummary();
  clout << "betaflow-done" << std::endl;
  return 0;
}
