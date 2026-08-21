/*  mcChannel3d — the mc_channel case through OpenLB's D3Q7 ADE lattice.
 *
 *  betaFlow Tier-1 exam for OpenLB (1.9): the molecular-communications
 *  channel impulse response of Hofmann et al. 2024 (Table-1 geometry,
 *  Pe = 200) as an Eulerian scalar. A slug of passive scalar is released
 *  in a cylinder with a PRESCRIBED analytic Poiseuille advection field —
 *  frozen-field discipline, matching the Langevin and OpenFOAM particle
 *  legs, so only the scalar transport equation is on trial. The CIR is
 *  window-integrated concentration over total mass, written as CSV for
 *  betaflow/runners/openlb.py to parse.
 *
 *  Structure adapted from examples/advectionDiffusionReaction/
 *  advectionDiffusion3d (Simonis, Frank & Krause setup); geometry and
 *  wall handling from microMixer3d (boundary::set<boundary::BounceBack>).
 *
 *  Known unresolved item this app MEASURES rather than assumes: the
 *  bounce-back wall position (effective radius a_eff = a + O(dx)) —
 *  declared UNRESOLVED in betaflow/analytic/lattice_boltzmann.py.
 *
 *  CLI (all optional):
 *    --resolution N   cells per RADIUS               (default 15)
 *    --tau X          ADE relaxation time            (default 0.6)
 *    --horizon X      end time in units of t2_max    (default 6.5)
 *    --outputs N      CSV rows to write              (default 160)
 *    --outdir PATH    output directory               (default ./tmp/)
 */

#include <olb.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

using namespace olb;
using namespace olb::names;

using MyCase = Case<
  AdvectionDiffusion, Lattice<double, descriptors::D3Q7<descriptors::VELOCITY>>
>;
using T = MyCase::value_t;
using DESCRIPTOR = MyCase::descriptor_t_of<AdvectionDiffusion>;

// Hofmann et al. 2024 Table 1 + betaFlow's dimensional split of Pe = 200
// (betaflow/cases/mc_channel.yaml). Fixed here on purpose: the runner
// varies numerics only, never physics.
static constexpr T RADIUS      = 200e-6;
static constexpr T U_MEAN      = 1.5e-3;
static constexpr T DIFFUSIVITY = 1.5e-9;
static constexpr T CX          = 100e-6;
static constexpr T DBAR[3]     = {150e-6, 750e-6, 1550e-6};
static constexpr T CS2_D3Q7    = 0.25;   // source-confirmed, descriptor cs2<3,7>

// Prescribed advection, returned in LATTICE units. Two sources:
//   analytic (default)  the exact Poiseuille parabola;
//   --profile FILE      a SOLVED axisymmetric profile from the fluid stage
//                       (CSV rows "y_over_a, u_x_phys", the exact format
//                       openlb_cases/pipeFlow3d writes) — this is the
//                       COUPLED channel scenario, staged: the flow is
//                       steady, so converging the fluid first and running
//                       the scalar on the solved field is physically
//                       identical to per-step coupling, and it decouples
//                       the two lattices' time steps. That matters here:
//                       Sc = nu/D ~ 667, so a SHARED dt would force
//                       tau_fluid ~ 2.9 (recorded as a scenario-design
//                       fact; staging avoids it).
class PoiseuilleLatticeVelocity : public AnalyticalF3D<T,T> {
  T _convVel;
  std::vector<T> _r, _u;   // solved profile table over |y|/a, empty = analytic
public:
  explicit PoiseuilleLatticeVelocity(T convVel, const std::string& profile)
    : AnalyticalF3D<T,T>(3), _convVel(convVel) {
    if (!profile.empty()) {
      std::ifstream f(profile);
      std::string line;
      while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        const auto comma = line.find(',');
        _r.push_back(std::fabs(std::atof(line.substr(0, comma).c_str())));
        _u.push_back(std::atof(line.substr(comma + 1).c_str()));
      }
      // sort by |r| for interpolation (profile file runs -a..a)
      std::vector<std::size_t> idx(_r.size());
      for (std::size_t i = 0; i < idx.size(); ++i) { idx[i] = i; }
      std::sort(idx.begin(), idx.end(),
                [&](std::size_t i, std::size_t j){ return _r[i] < _r[j]; });
      std::vector<T> rs, us;
      for (auto i : idx) { rs.push_back(_r[i]); us.push_back(_u[i]); }
      _r = rs; _u = us;
    }
  }
  bool operator()(T out[], const T in[]) override {
    const T r = std::sqrt(in[1]*in[1] + in[2]*in[2]) / RADIUS;
    T u;
    if (_r.empty()) {
      u = 2.0 * U_MEAN * util::max(T(0), T(1) - r*r);
    } else if (r >= _r.back()) {
      u = T(0);
    } else {
      auto hi = std::upper_bound(_r.begin(), _r.end(), r) - _r.begin();
      if (hi == 0) { u = _u[0]; }
      else {
        const T w = (r - _r[hi-1]) / (_r[hi] - _r[hi-1]);
        u = _u[hi-1] * (1 - w) + _u[hi] * w;
      }
      u = util::max(T(0), u);
    }
    out[0] = u / _convVel;
    out[1] = T(0);
    out[2] = T(0);
    return true;
  }
};

// Uniform slug of scalar, axial extent [x0 - w/2, x0 + w/2], INSIDE the
// cylinder only. The radial guard matters for mass conservation: wall and
// corner cells (material 2) must start at zero, or their default unit
// density bleeds into the bulk through the bounce-back — measured as a
// 66% mass drift before this guard existed.
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
  OstreamManager clout(std::cout, "mcChannel3d");
  initialize(&argc, &argv);

  const int  res     = int(argOpt(argc, argv, "--resolution", 15));
  const T    tau     = argOpt(argc, argv, "--tau", 0.6);
  const T    horizon = argOpt(argc, argv, "--horizon", 6.5);
  const int  outputs = int(argOpt(argc, argv, "--outputs", 160));
  const std::string outdir = argStr(argc, argv, "--outdir", "./tmp/");
  const std::string profile = argStr(argc, argv, "--profile", "");
  singleton::directories().setOutputDir(outdir);

  const T dx = RADIUS / T(res);
  // dt from the REQUESTED tau: D_lat = D dt/dx^2 = cs2 (tau - 1/2).
  const T dt = (tau - 0.5) * CS2_D3Q7 * dx * dx / DIFFUSIVITY;

  const T t2max = (DBAR[2] + CX/2.0) / (2.0 * U_MEAN);
  const T tMax  = horizon * t2max;
  const int iTmax = int(std::ceil(tMax / dt));
  const int statIter = util::max(1, iTmax / outputs);

  // Slug near the inlet; cyclic axial ends, so the domain must be long
  // enough that the fastest scalar never wraps within tMax.
  const T slugW = 2.0 * dx;
  const T x0 = 4.0 * dx + slugW;
  const T drift = 2.0 * U_MEAN * tMax;
  const T spread = 4.0 * std::sqrt(2.0 * DIFFUSIVITY * tMax);
  // Domain covers the drift AND the farthest receiver window, whichever is
  // longer — short-horizon runs must still contain every window.
  const T length = x0 + util::max(drift + spread, DBAR[2] + CX) + 10.0 * dx;

  const T pad = 2.0 * dx;
  Vector<T,3> extent(length, 2.0*(RADIUS + pad), 2.0*(RADIUS + pad));
  Vector<T,3> origin(T(0), -(RADIUS + pad), -(RADIUS + pad));
  IndicatorCuboid3D<T> box(extent, origin);

#ifdef PARALLEL_MODE_MPI
  const int noOfCuboids = singleton::mpi().getSize();
#else
  const int noOfCuboids = 1;
#endif
  Mesh<T,MyCase::d> mesh(box, dx, noOfCuboids);
  mesh.setOverlap(2);
  mesh.getCuboidDecomposition().setPeriodicity({true, false, false});

  MyCase::ParametersD params;
  MyCase myCase(params, mesh);

  auto& geometry = myCase.getGeometry();
  geometry.rename(0, 2);
  IndicatorCylinder3D<T> pipe(Vector<T,3>(-2.0*dx, T(0), T(0)),
                              Vector<T,3>(length + 2.0*dx, T(0), T(0)),
                              RADIUS);
  geometry.rename(2, 1, pipe);
  geometry.communicate();
  geometry.print();

  auto& lattice = myCase.getLattice(AdvectionDiffusion{});
  lattice.setUnitConverter<AdeUnitConverter<T,DESCRIPTOR>>(
    dx, dt, RADIUS, 2.0*U_MEAN, DIFFUSIVITY, T(1));
  const auto& converter = lattice.getUnitConverter();
  converter.print();

  dynamics::set<AdvectionDiffusionBGKdynamics>(
    lattice, geometry.getMaterialIndicator({1}));
  // Stock BounceBack, with an instrumentation finding the runner records:
  // NO density functor on bounce-back cells is mass accounting. The stock
  // momenta::FixedDensity reads 1 whatever the populations hold (measured:
  // a fake 20000-unit total over the shell), and BounceBackBulkDensity
  // reads the Revert collision's period-2 population cycle (measured: the
  // total alternating between two exact values). So the CIR below uses
  // BULK-ONLY sums for numerator AND denominator — consistent by
  // construction — and the mass parked in wall cells mid-flight appears as
  // a slow, mechanism-named decline of the bulk total, recorded not hidden.
  boundary::set<boundary::BounceBack>(lattice, geometry, 2);

  PoiseuilleLatticeVelocity uF(converter.getConversionFactorVelocity(), profile);
  SlugInit slugF(x0, slugW);
  auto everything = geometry.getMaterialIndicator({1, 2});
  fields::set<descriptors::VELOCITY>(lattice, everything, uF);
  // iniEquilibrium, NOT momenta::setDensity: the latter never reaches the
  // BounceBack wall cells, whose default unit density then bleeds into the
  // bulk (measured: mass grew 2.5x before this line was right; the pattern
  // is microMixer3d's, whose init indicator also includes the walls).
  // Walls are zeroed EXPLICITLY: a slug edge cell whose centre falls in the
  // wall ring would otherwise park mass that later streams into the bulk
  // (measured as the residual 3.7% drift).
  AnalyticalConst3D<T,T> zeroRho(T(0));
  AnalyticalConst3D<T,T> zeroU(T(0), T(0), T(0));
  lattice.iniEquilibrium(geometry.getMaterialIndicator({1}), slugF, uF);
  lattice.iniEquilibrium(geometry.getMaterialIndicator({2}), zeroRho, zeroU);
  lattice.setParameter<descriptors::OMEGA>(
    converter.getLatticeAdeRelaxationFrequency());
  lattice.initialize();

  // Provenance the runner parses. omega must recover the requested tau.
  clout << "betaflow-provenance"
        << " velocity_source=" << (profile.empty() ? std::string("analytic") : profile)
        << " tau_requested=" << tau
        << " omega=" << converter.getLatticeAdeRelaxationFrequency()
        << " dx=" << dx << " dt=" << dt
        << " uLatCentre=" << 2.0*U_MEAN / converter.getConversionFactorVelocity()
        << " iTmax=" << iTmax << " statIter=" << statIter
        << " length=" << length << " x0=" << x0 << " slugW=" << slugW
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
      // Bulk-only total (see the wall-instrumentation note above): the
      // window sums are also bulk-only, so the CIR is the fraction of the
      // mass currently IN the bulk — exact under that stated measure. The
      // parked-in-wall share shows up as this column's slow decline.
      SuperSum3D<T,T> totalSum(rho, geometry, 1);
      totalSum(total, tmp);

      T cir[3];
      for (int w = 0; w < 3; ++w) {
        SuperSum3D<T,T> winSum(
          std::unique_ptr<SuperF3D<T,T>>(
            new SuperLatticeDensity3D<T,DESCRIPTOR>(lattice)),
          std::unique_ptr<SuperIndicatorF3D<T>>(
            new SuperIndicatorFfromIndicatorF3D<T>(
              std::unique_ptr<IndicatorF3D<T>>(
                new IndicatorCylinder3D<T>(
                  Vector<T,3>(x0 + DBAR[w] - CX/2.0, T(0), T(0)),
                  Vector<T,3>(x0 + DBAR[w] + CX/2.0, T(0), T(0)),
                  RADIUS)),  // bulk-only, consistent with the total
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
