/*  pipeFlow3d — betaFlow's momentum rung for OpenLB (1.9).
 *
 *  The FLUID exam: steady forced Poiseuille flow in the circular pipe of
 *  betaflow/cases/pipe_poiseuille_steady.yaml (radius 1 m, bulk-Re 100),
 *  on the D3Q19 lattice with ForcedBGKdynamics — the same case, exact
 *  solution, and metrics OpenFOAM already passed, now applied to OpenLB's
 *  Navier-Stokes solver. Adapted from examples/laminar/poiseuille3d
 *  (FORCED path: periodic x, constant body force, warm start from the
 *  exact profile, ValueTracer convergence on average energy).
 *
 *  THE QUESTION THIS APP EXISTS TO MEASURE, with the predictions written
 *  before the first run (they are judged in tests/test_openlb.py and
 *  recorded either way):
 *
 *  Where does the bounce-back wall actually sit? The reflection rule
 *  places the no-slip plane BETWEEN grid points, and on a staircase
 *  cylinder the effective radius a_eff differs from the geometric a by
 *  some fraction of dx — declared UNRESOLVED in
 *  betaflow/analytic/lattice_boltzmann.py until now.
 *    H-bb:      a_eff = a + c dx with c of order one and positive (the
 *               wall sits OUTSIDE the last fluid node), and the staircase
 *               drags the velocity-error convergence toward FIRST order.
 *    H-bouzidi: the control. Bouzidi interpolation honours the true wall
 *               distance per link, so |a_eff - a| = O(dx^2) and the error
 *               converges at ~SECOND order.
 *
 *  The applied force is the exact Poiseuille relation for a wall AT r = a
 *  (f = 4 nu u_max / a^2), so any wall-position shift shows up directly:
 *  the realised profile is the parabola of a_eff, wider and faster than
 *  the reference. The profile is sampled along a diameter and written as
 *  CSV for betaflow/runners/openlb.py, which fits a_eff and reuses the
 *  case's own L2 metric.
 *
 *  CLI (all optional):
 *    --resolution N   cells per DIAMETER          (default 21)
 *    --tau X          BGK relaxation time         (default 0.55; the
 *                     runner picks it from a target lattice velocity —
 *                     stability pins tau toward 1/2 here exactly as it
 *                     did for the ADE lattice)
 *    --wall S         "bb" or "bouzidi"           (default bb)
 *    --maxt X         cap on physical time [s]    (default 120)
 *    --outdir PATH                                (default ./tmp/)
 */

#include <olb.h>

#include <cstring>
#include <fstream>

using namespace olb;
using namespace olb::names;

using MyCase = Case<
  NavierStokes, Lattice<double, descriptors::D3Q19<descriptors::FORCE>>
>;
using T = MyCase::value_t;
using DESCRIPTOR = MyCase::descriptor_t_of<NavierStokes>;

// Defaults: betaflow/cases/pipe_poiseuille_steady.yaml (radius 1 m, bulk
// Re 100 -> u_max 1, nu 0.01). Overridable via --radius/--umax/--nu so the
// COUPLED channel scenario can run the same app at the microfluidic
// point (a = 200 um, u_max = 3 mm/s, water nu = 1e-6: Re = 0.6). The
// runner records both Reynolds conventions; this file works in
// (u_max, nu) only.
static T RADIUS = 1.0;
static T U_MAX  = 1.0;
static T NU     = 0.01;

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
  OstreamManager clout(std::cout, "pipeFlow3d");
  initialize(&argc, &argv);

  const int  res  = int(argOpt(argc, argv, "--resolution", 21));
  const T    tau  = argOpt(argc, argv, "--tau", 0.55);
  const T    maxT = argOpt(argc, argv, "--maxt", 120.0);
  const std::string wall = argStr(argc, argv, "--wall", "bb");
  const std::string outdir = argStr(argc, argv, "--outdir", "./tmp/");
  RADIUS = argOpt(argc, argv, "--radius", RADIUS);
  U_MAX  = argOpt(argc, argv, "--umax", U_MAX);
  NU     = argOpt(argc, argv, "--nu", NU);
  singleton::directories().setOutputDir(outdir);

  const T diameter = 2.0 * RADIUS;
  const T dx = diameter / T(res);
  // Short periodic pipe: the forced solution is x-invariant.
  const T length = 16.0 * dx;

  // Geometry construction follows the shipped example's FORCED path
  // exactly (axis at y = z = RADIUS, x-margins so the periodic faces stay
  // open, one wall layer around the cylinder).
  Vector<T,3> center0(0.0, RADIUS, RADIUS);
  Vector<T,3> center1(length + 0.5 * dx, RADIUS, RADIUS);
  IndicatorCylinder3D<T> meshPipe(center0, center1, RADIUS);
  IndicatorLayer3D<T> extendedDomain(meshPipe, dx);

#ifdef PARALLEL_MODE_MPI
  const int noOfCuboids = singleton::mpi().getSize();
#else
  const int noOfCuboids = 1;
#endif
  Mesh<T,MyCase::d> mesh(extendedDomain, dx, noOfCuboids);
  mesh.setOverlap(3);
  mesh.getCuboidDecomposition().setPeriodicity({true, false, false});

  MyCase::ParametersD params;
  MyCase myCase(params, mesh);

  auto& geometry = myCase.getGeometry();
  Vector<T,3> g0(-0.2 * dx - 3.0 * dx, RADIUS, RADIUS);
  Vector<T,3> g1(length + 3.0 * dx, RADIUS, RADIUS);
  IndicatorCylinder3D<T> pipe(g0, g1, RADIUS);
  geometry.rename(0, 2);
  geometry.rename(2, 1, pipe);
  geometry.clean();
  geometry.innerClean();
  geometry.checkForErrors();
  geometry.print();

  auto& lattice = myCase.getLattice(NavierStokes{});
  lattice.setUnitConverter<UnitConverterFromResolutionAndRelaxationTime<T,DESCRIPTOR>>(
    res, tau, diameter, U_MAX, NU, T(1));
  const auto& converter = lattice.getUnitConverter();
  converter.print();

  dynamics::set<ForcedBGKdynamics>(lattice, geometry, 1);

  // The wall under test vs the control.
  if (wall == "bouzidi") {
    Vector<T,3> b0(g0);  b0[0] -= 0.5 * dx;
    Vector<T,3> b1(g1);  b1[0] += 0.5 * dx;
    IndicatorCylinder3D<T> bouzidiPipe(b0, b1, RADIUS);
    setBouzidiBoundary<T, DESCRIPTOR, BouzidiPostProcessor>(
      lattice, geometry, 2, bouzidiPipe);
  } else {
    boundary::set<boundary::BounceBack>(lattice, geometry, 2);
  }

  // Constant body force: the exact Poiseuille relation for a wall AT
  // r = a, in lattice units — identical to the shipped example.
  const T dLat = converter.getLatticeLength(diameter);
  Vector<T,3> forceLat(T(0));
  forceLat[0] = 4.0 * converter.getLatticeViscosity()
              * converter.getCharLatticeVelocity() / (dLat * dLat / 4.0);
  fields::set<descriptors::FORCE>(
    lattice, geometry.getMaterialIndicator({1, 2}), forceLat);

  // Warm start from the exact profile: convergence then measures the
  // DIFFERENCE between the exact solution and the lattice's own steady
  // state, which is precisely the quantity under test.
  std::vector<T> axisPoint = {length, RADIUS, RADIUS};
  std::vector<T> axisDir = {1, 0, 0};
  CirclePoiseuille3D<T> uInit(axisPoint, axisDir, U_MAX, RADIUS);
  momenta::setVelocity(lattice, geometry.getMaterialIndicator({1, 2}), uInit);
  lattice.setParameter<descriptors::OMEGA>(
    converter.getLatticeRelaxationFrequency());
  lattice.initialize();

  clout << "betaflow-provenance"
        << " radius=" << RADIUS << " umax=" << U_MAX << " nu=" << NU
        << " wall=" << wall
        << " tau_requested=" << tau
        << " omega=" << converter.getLatticeRelaxationFrequency()
        << " dx=" << dx
        << " dt=" << converter.getPhysDeltaT()
        << " uLatChar=" << converter.getCharLatticeVelocity()
        << " forceLat=" << forceLat[0]
        << " nuLat=" << converter.getLatticeViscosity()
        << std::endl;

  const size_t iTmax = converter.getLatticeTime(maxT);
  const size_t iTcheck = converter.getLatticeTime(0.5);
  util::Timer<T> timer(iTmax, geometry.getStatistics().getNvoxel());
  util::ValueTracer<T> converge(iTcheck, 1e-9);
  timer.start();

  size_t iT = 0;
  for (; iT < iTmax; ++iT) {
    if (converge.hasConverged()) {
      clout << "converged at iT=" << iT << std::endl;
      break;
    }
    lattice.collideAndStream();
    converge.takeValue(lattice.getStatistics().getAverageEnergy(), false);
    if (iT % (iTcheck * 20) == 0) {
      timer.update(iT);
      timer.printStep();
    }
  }
  timer.stop();

  // Sample u_x along a diameter (y-direction, through the axis, at the
  // axial midpoint), interpolated, 201 fixed stations. The CSV column
  // "y" is the wall-normal coordinate relative to the axis, matching the
  // convention every betaflow fluid runner returns.
  lattice.setProcessingContext(ProcessingContext::Evaluation);
  SuperLatticePhysVelocity3D<T,DESCRIPTOR> uPhys(lattice, converter);
  AnalyticalFfromSuperF3D<T> uInterp(uPhys, true);

  std::ofstream csv(outdir + "profile.csv");
  csv.precision(12);
  csv << "# y_over_a, u_x_phys\n";
  const int nSample = 201;
  for (int k = 0; k < nSample; ++k) {
    const T yLoc = -RADIUS + (2.0 * RADIUS) * T(k) / T(nSample - 1);
    // Stay a hair inside the geometric wall so interpolation has support.
    const T yEval = util::max(-RADIUS + 0.51 * dx,
                              util::min(RADIUS - 0.51 * dx, yLoc));
    T point[3] = {0.5 * length, RADIUS + yEval, RADIUS};
    T u[3] = {T(0), T(0), T(0)};
    uInterp(u, point);
    csv << yEval / RADIUS << ", " << u[0] << "\n";
  }
  csv.close();

  clout << "betaflow-done steps=" << iT << std::endl;
  return 0;
}
