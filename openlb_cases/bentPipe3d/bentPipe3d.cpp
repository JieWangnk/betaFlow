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

using MyCase = Case<
  AdvectionDiffusion, Lattice<double, descriptors::D3Q7<descriptors::VELOCITY>>
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
    T r2, tx, ty;
    const T phi = (_f.angle > T(0)) ? _f.phiOf(in) : T(-1);
    if (_f.angle <= T(0) || phi <= T(0) || in[0] <= _f.xb) {
      // mother frame (also the whole domain at angle 0)
      if (_f.angle > T(0) && phi > T(0) && in[0] > _f.xb) {
        // unreachable guard; kept for clarity
      }
      r2 = in[1]*in[1] + in[2]*in[2];
      tx = T(1); ty = T(0);
    } else if (phi < _f.angle) {
      // bend: radial distance from the arc centreline
      const T wx = in[0] - _f.C[0], wy = in[1] - _f.C[1];
      const T dR = std::sqrt(wx*wx + wy*wy) - _f.R;
      r2 = dR*dR + in[2]*in[2];
      tx = std::cos(phi); ty = std::sin(phi);
    } else {
      // daughter frame
      const Vector<T,3> q(in[0] - _f.B1[0], in[1] - _f.B1[1], in[2]);
      const T s = q * _f.d2;
      const Vector<T,3> rad = q - _f.d2 * s;
      r2 = rad * rad;
      tx = _f.d2[0]; ty = _f.d2[1];
    }
    const T u = 2.0 * U_MEAN
                * util::max(T(0), T(1) - r2 / (RADIUS*RADIUS)) / _convVel;
    out[0] = u * tx;
    out[1] = u * ty;
    out[2] = T(0);
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
  geometry.communicate();
  geometry.print();

  auto& lattice = myCase.getLattice(AdvectionDiffusion{});
  lattice.setUnitConverter<AdeUnitConverter<T,DESCRIPTOR>>(
    dx, dt, RADIUS, 2.0*U_MEAN, DIFFUSIVITY, T(1));
  const auto& converter = lattice.getUnitConverter();
  converter.print();

  if (dyn == "trt") {
    dynamics::set<AdvectionDiffusionTRTdynamics>(
      lattice, geometry.getMaterialIndicator({1}));
  } else if (dyn == "bgk2") {
    dynamics::set<AdeSecondOrderBGKdynamics>(
      lattice, geometry.getMaterialIndicator({1}));
  } else {
    dynamics::set<AdvectionDiffusionBGKdynamics>(
      lattice, geometry.getMaterialIndicator({1}));
  }
  if (wall == "bouzidi") {
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
  auto everything = geometry.getMaterialIndicator({1, 2});
  fields::set<descriptors::VELOCITY>(lattice, everything, uF);
  AnalyticalConst3D<T,T> zeroRho(T(0));
  AnalyticalConst3D<T,T> zeroU(T(0), T(0), T(0));
  lattice.iniEquilibrium(geometry.getMaterialIndicator({1}), slugF, uF);
  lattice.iniEquilibrium(geometry.getMaterialIndicator({2}), zeroRho, zeroU);
  if (dyn == "trt") {
    // OMEGA relaxes the EVEN sector (free choice); MAGIC places the ODD
    // rate, which carries D: Lambda = (tau_even - 1/2)(tau_ade - 1/2).
    lattice.setParameter<descriptors::OMEGA>(T(1) / tauEven);
    lattice.setParameter<collision::TRT::MAGIC>(
      (tauEven - 0.5) * (tau - 0.5));
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
      SuperSum3D<T,T> totalSum(rho, geometry, 1);
      totalSum(total, tmp);

      T cir[3];
      for (int w = 0; w < 3; ++w) {
        // Window at PATH distance DBAR[w] from the slug centre. Window 1
        // lies in the straight mother (path < sbend) — the in-run
        // control; windows 2 and 3 lie in the outgoing straight section
        // (path > sbend + arclen). No window intersects the bend.
        Vector<T,3> w0, w1;
        if (DBAR[w] + CX/2.0 <= sBend) {
          w0 = Vector<T,3>(x0 + DBAR[w] - CX/2.0, T(0), T(0));
          w1 = Vector<T,3>(x0 + DBAR[w] + CX/2.0, T(0), T(0));
        } else {
          const T sOut = DBAR[w] - sBend - arcLen;   // path past bend exit
          w0 = f.B1 + f.d2 * (sOut - CX/2.0);
          w1 = f.B1 + f.d2 * (sOut + CX/2.0);
        }
        SuperSum3D<T,T> winSum(
          std::unique_ptr<SuperF3D<T,T>>(
            new SuperLatticeDensity3D<T,DESCRIPTOR>(lattice)),
          std::unique_ptr<SuperIndicatorF3D<T>>(
            new SuperIndicatorFfromIndicatorF3D<T>(
              std::unique_ptr<IndicatorF3D<T>>(
                new IndicatorCylinder3D<T>(w0, w1, RADIUS)),
              geometry)));
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
