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
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_map>

using namespace olb;
using namespace olb::names;

#include "bentGeometry.h"
using bent::RADIUS;
using bent::U_MEAN;
using bent::DIFFUSIVITY;
using bent::CX;
using bent::DBAR;
using BendFrame = bent::BendFrame;

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

// Region-wise analytic Poiseuille in LATTICE units, direction continuous
// everywhere. Angle 0 reduces exactly to the straight analytic field.
class BentPoiseuilleVelocity : public AnalyticalF3D<T,T> {
  T _convVel;
  BendFrame _f;
public:
  BentPoiseuilleVelocity(T convVel, BendFrame f)
    : AnalyticalF3D<T,T>(3), _convVel(convVel), _f(f) {}
  bool operator()(T out[], const T in[]) override {
    T uPhys[3];
    _f.analyticU(in, uPhys);
    out[0] = uPhys[0] / _convVel;
    out[1] = uPhys[1] / _convVel;
    out[2] = uPhys[2] / _convVel;
    return true;
  }
};

// Prescribed velocity from a SOLVED field (bentFlow3d's ufield.csv):
// nearest-cell lookup keyed on round(2 x / dx) per axis — the two apps
// share box, origin and dx, so the file's sample points are this app's
// cell centres. Cells absent from the file (walls, beyond caps) get zero.
class SolvedFieldVelocity : public AnalyticalF3D<T,T> {
  T _convVel, _dx;
  std::unordered_map<long long, std::array<T,3>> _u;
  long long key(const T in[]) const {
    const long long a = llround(2.0*in[0]/_dx) + 4096;
    const long long b = llround(2.0*in[1]/_dx) + 4096;
    const long long c = llround(2.0*in[2]/_dx) + 4096;
    return (a << 28) | (b << 14) | c;
  }
public:
  SolvedFieldVelocity(T convVel, T dx, const std::string& file)
    : AnalyticalF3D<T,T>(3), _convVel(convVel), _dx(dx) {
    std::ifstream fin(file);
    std::string line;
    while (std::getline(fin, line)) {
      if (line.empty() || line[0] == '#') { continue; }
      std::array<T,4+2> v{};
      std::size_t pos = 0;
      for (int k = 0; k < 6; ++k) {
        const auto comma = line.find(',', pos);
        v[k] = std::atof(line.substr(pos, comma - pos).c_str());
        pos = (comma == std::string::npos) ? line.size() : comma + 1;
      }
      const T pt[3] = {v[0], v[1], v[2]};
      _u[key(pt)] = {v[3], v[4], v[5]};
    }
  }
  std::size_t size() const { return _u.size(); }
  bool operator()(T out[], const T in[]) override {
    const auto it = _u.find(key(in));
    if (it == _u.end()) {
      out[0] = out[1] = out[2] = T(0);
    } else {
      out[0] = it->second[0] / _convVel;
      out[1] = it->second[1] / _convVel;
      out[2] = it->second[2] / _convVel;
    }
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
  // --ufield FILE: advect on the SOLVED flow (bentFlow3d output) instead
  // of the region-wise analytic field — the solved-flow leg of the bend,
  // staged exactly as the straight coupled model staged it.
  const std::string ufield = argStr(argc, argv, "--ufield", "");
  const std::string outdir = argStr(argc, argv, "--outdir", "./tmp/");
  singleton::directories().setOutputDir(outdir);

  // ALL geometry/layout from the shared header (bentGeometry.h) — the
  // fluid app computes the identical layout from the same inputs, so the
  // two lattices agree cell-for-cell. Slug half-width 1.01 dx: the
  // deterministic 3-slice slug (the edge-rounding finding in the G4
  // record).
  const bent::BentLayout L(res, horizon, angleDeg, rBend, sBend);
  const T dx = L.dx;
  const T dt = (tau - 0.5) * CS2_D3Q7 * dx * dx / DIFFUSIVITY;
  const int iTmax = int(std::ceil(L.tMax / dt));
  const int statIter = util::max(1, iTmax / outputs);
  const T slugHalf = 1.01 * dx;
  const T slugW = L.slugW;
  const T x0 = L.x0;
  const T pathLen = L.pathLen;
  const BendFrame& f = L.f;
  const T arcLen = L.arcLen;
  const T daughterLen = L.daughterLen;
  const Vector<T,3> pEnd = L.pEnd;

  Vector<T,3> extent(L.xMax, L.yMax - L.yMin, 2.0*(RADIUS + L.pad));
  Vector<T,3> origin(T(0), L.yMin, -(RADIUS + L.pad));
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
  if (f.angle > T(0)) {
    constexpr int NSEG = 8;
    for (int k = 0; k < NSEG; ++k) {
      const T p0 = f.angle * T(k) / NSEG, p1 = f.angle * T(k + 1) / NSEG;
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
    SolidFraction epsF(f, dx, 2.0*dx, f.xb + arcLen + daughterLen);
    fields::set<SOLID_FRACTION>(
      lattice, geometry.getMaterialIndicator({1, 2, 11, 12, 13}), epsF);
  } else if (wall == "bouzidi") {
    auto surf = bent::boreSurface(L);
    setBouzidiBoundary<T, DESCRIPTOR, BouzidiPostProcessor>(
      lattice, geometry, 2, *surf);
  } else {
    boundary::set<boundary::BounceBack>(lattice, geometry, 2);
  }

  BentPoiseuilleVelocity uAnalytic(converter.getConversionFactorVelocity(), f);
  std::unique_ptr<SolvedFieldVelocity> uSolved;
  if (!ufield.empty()) {
    uSolved.reset(new SolvedFieldVelocity(
      converter.getConversionFactorVelocity(), dx, ufield));
    // HARD GUARD against a silent grid-convention mismatch (measured
    // once: every lookup missed, the field read zero, the scalar crawled
    // at lag +51 t2 with nothing to flag it): the loaded field at the
    // slug centre must carry roughly the centreline speed.
    T probe[3] = {T(0), T(0), T(0)};
    const T at[3] = {x0, T(0), T(0)};
    (*uSolved)(probe, at);
    const T expect = 2.0 * U_MEAN / converter.getConversionFactorVelocity();
    if (std::fabs(probe[0]) < 0.3 * expect) {
      clout << "FATAL: solved field reads " << probe[0]
            << " lattice units at the slug centre against an expected ~"
            << expect << " - grid-convention mismatch with " << ufield
            << std::endl;
      return 1;
    }
    clout << "solved field loaded: " << uSolved->size() << " cells from "
          << ufield << " (centre u_lat=" << probe[0] << ")" << std::endl;
  }
  AnalyticalF3D<T,T>& uF = ufield.empty()
    ? static_cast<AnalyticalF3D<T,T>&>(uAnalytic)
    : static_cast<AnalyticalF3D<T,T>&>(*uSolved);
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
        << " velocity_source="
        << (ufield.empty() ? std::string("regionwise-analytic-arc") : ufield)
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
      T total[2] = {T(0), T(0)};
      int tmp[1] = {0};
      // Bulk-only by construction: material sums over {1, 11, 12, 13}.
      total[0] = T(0);
      for (int mat : {1, 11, 12, 13}) {
        // SuperSum3D writes the sum AND the cell count: two outputs
        // (a one-element array here smashed the junction app's stack).
        T part[2] = {T(0), T(0)};
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
        T out[2] = {T(0), T(0)};
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
