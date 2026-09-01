/*  bentFlow3d — the SOLVED flow through the bent pipe (G4's fluid stage).
 *
 *  D3Q19 steady flow through the identical geometry the scalar app runs
 *  (bentGeometry.h is the single shared definition): developed Poiseuille
 *  fed at the mother inlet, pressure outlet at the daughter end, Bouzidi
 *  walls on the true bore surface (the wall the wall-position measurement
 *  chose for fluid lattices: shift ~ dx^2, order 2.1). Warm-started from
 *  the region-wise analytic bent field, so convergence measures the
 *  DIFFERENCE between the analytic prescription and the lattice's own
 *  steady solution — precisely the quantity the solved-vs-prescribed
 *  scalar comparison then propagates.
 *
 *  The scalar app consumes the output via --ufield: u is sampled at every
 *  grid cell centre within a cell of the bore (the two apps share box,
 *  origin and dx, so the sample points ARE the scalar's cells) and written
 *  as "x, y, z, ux, uy, uz" in physical units.
 *
 *  Gates the runner computes from the outputs (pre-registration G2/G7):
 *  flux balance between mother and daughter cross-sections; profile L2
 *  against the parabola in the straight mother and mid-daughter; P2's
 *  no-recirculation check (minimum tangential velocity in the bend).
 *
 *  CLI: --resolution (cells per RADIUS, default 12), --tau (default 0.8),
 *  --angle --rbend --sbend --horizon as the scalar app (layout must
 *  match), --maxt (fluid time cap [s], default 0.4), --outdir.
 */

#include <olb.h>

#include <cmath>
#include <cstring>
#include <fstream>

using namespace olb;
using namespace olb::names;

using MyCase = Case<
  NavierStokes, Lattice<double, descriptors::D3Q19<>>
>;
using T = MyCase::value_t;
using DESCRIPTOR = MyCase::descriptor_t_of<NavierStokes>;

#include "../bentPipe3d/bentGeometry.h"
using bent::RADIUS;
using bent::U_MEAN;
using bent::NU_WATER;

// Analytic bent Poiseuille in PHYSICAL units (warm start + inlet feed).
class BentAnalyticU : public AnalyticalF3D<T,T> {
  const bent::BendFrame _f;
public:
  explicit BentAnalyticU(const bent::BendFrame& f)
    : AnalyticalF3D<T,T>(3), _f(f) {}
  bool operator()(T out[], const T in[]) override {
    _f.analyticU(in, out);
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
  OstreamManager clout(std::cout, "bentFlow3d");
  initialize(&argc, &argv);

  const int  res     = int(argOpt(argc, argv, "--resolution", 12));
  const T    tau     = argOpt(argc, argv, "--tau", 0.8);
  const T    angleDeg = argOpt(argc, argv, "--angle", 0.0);
  const T    rBend   = argOpt(argc, argv, "--rbend", 400e-6);
  const T    sBend   = argOpt(argc, argv, "--sbend", 245e-6);
  const T    horizon = argOpt(argc, argv, "--horizon", 6.5);
  const T    maxt    = argOpt(argc, argv, "--maxt", 0.4);
  const std::string outdir = argStr(argc, argv, "--outdir", "./tmp/");
  singleton::directories().setOutputDir(outdir);

  const bent::BentLayout L(res, horizon, angleDeg, rBend, sBend);
  const bent::BendFrame& f = L.f;
  const T dx = L.dx;

  Vector<T,3> extent(L.xMax, L.yMax - L.yMin, 2.0*(RADIUS + L.pad));
  Vector<T,3> origin(T(0), L.yMin, -(RADIUS + L.pad));
  IndicatorCuboid3D<T> box(extent, origin);

#ifdef PARALLEL_MODE_MPI
  const int noOfCuboids = singleton::mpi().getSize();
#else
  const int noOfCuboids = 1;
#endif
  Mesh<T,MyCase::d> mesh(box, dx, noOfCuboids);
  mesh.setOverlap(3);
  mesh.getCuboidDecomposition().setPeriodicity({false, false, false});

  MyCase::ParametersD params;
  MyCase myCase(params, mesh);

  auto& geometry = myCase.getGeometry();
  geometry.rename(0, 2);
  // Bore staircase: identical construction to the scalar app.
  IndicatorCylinder3D<T> mother(Vector<T,3>(2.0*dx, T(0), T(0)),
                                Vector<T,3>(f.xb, T(0), T(0)), RADIUS);
  geometry.rename(2, 1, mother);
  if (f.angle > T(0)) {
    constexpr int NSEG = 8;
    for (int k = 0; k < NSEG; ++k) {
      const T p0 = f.angle * T(k) / NSEG, p1 = f.angle * T(k + 1) / NSEG;
      const Vector<T,3> a0 = f.C
        + Vector<T,3>(f.R*std::sin(p0), -f.R*std::cos(p0), T(0));
      const Vector<T,3> a1 = f.C
        + Vector<T,3>(f.R*std::sin(p1), -f.R*std::cos(p1), T(0));
      IndicatorCylinder3D<T> seg(a0, a1, RADIUS);
      geometry.rename(2, 1, seg);
    }
    IndicatorCylinder3D<T> daughter(f.B1, L.pEnd, RADIUS);
    geometry.rename(2, 1, daughter);
  } else {
    IndicatorCylinder3D<T> rest(Vector<T,3>(f.xb, T(0), T(0)),
                                L.pEnd, RADIUS);
    geometry.rename(2, 1, rest);
  }
  // Inlet (3) at the mother start, outlet (4) at the daughter end: wall
  // cells adjacent to the bore inside the end discs.
  // Unconditional disc renames: inside these end discs the only
  // material-2 cells at r < a are the cap slices (the lateral wall sits
  // at r > a), so the discs select exactly the openings. The conditional
  // rename(2,3,1,ind) variant matched nothing here (measured: zero
  // material-3/4 cells, a dead inlet, fluxes ~1e-14).
  IndicatorCylinder3D<T> inflow(Vector<T,3>(T(0), T(0), T(0)),
                                Vector<T,3>(1.6*dx, T(0), T(0)),
                                RADIUS - 0.1*dx);
  geometry.rename(2, 3, inflow);
  IndicatorCylinder3D<T> outflow(L.pEnd + f.d2 * (0.4*dx),
                                 L.pEnd + f.d2 * (2.0*dx),
                                 RADIUS - 0.1*dx);
  geometry.rename(2, 4, outflow);
  geometry.clean();
  geometry.innerClean();
  geometry.checkForErrors();
  geometry.print();

  auto& lattice = myCase.getLattice(NavierStokes{});
  lattice.setUnitConverter<
    UnitConverterFromResolutionAndRelaxationTime<T,DESCRIPTOR>>(
    2*res, tau, 2.0*RADIUS, 2.0*U_MEAN, NU_WATER, T(1));
  const auto& converter = lattice.getUnitConverter();
  converter.print();

  dynamics::set<BGKdynamics>(lattice, geometry.getMaterialIndicator({1}));
  // Lateral wall: Bouzidi on the true surface, overshooting the open ends
  // by 4 dx so no spurious end cap crosses the inlet/outlet links.
  auto surf = bent::boreSurface(L, 4.0*dx);
  setBouzidiBoundary<T, DESCRIPTOR, BouzidiPostProcessor>(
    lattice, geometry, 2, *surf);
  boundary::set<boundary::InterpolatedVelocity>(lattice, geometry, 3);
  boundary::set<boundary::InterpolatedPressure>(lattice, geometry, 4);

  BentAnalyticU uAna(f);
  momenta::setVelocity(lattice, geometry.getMaterialIndicator({1, 3}),
                       uAna);
  lattice.setParameter<descriptors::OMEGA>(
    converter.getLatticeRelaxationFrequency());
  lattice.initialize();
  // The inlet Dirichlet value must be (re)defined AFTER initialize and
  // pushed to the compute context — the cylinder3d pattern; without the
  // push the InterpolatedVelocity boundary holds zero and the warm-started
  // flow drains (measured: fluxes ~1e-14 on the first smoke run).
  momenta::setVelocity(lattice, geometry.getMaterialIndicator({3}), uAna);
  lattice.setProcessingContext<
    Array<momenta::FixedVelocityMomentumGeneric::VELOCITY>>(
    ProcessingContext::Simulation);

  const int iTmax = int(std::ceil(maxt / converter.getPhysDeltaT()));
  const int iTcheck = util::max(1, iTmax / 200);
  util::ValueTracer<T> converge(iTcheck, 1e-11);

  clout << "betaflow-provenance"
        << " angle_deg=" << angleDeg
        << " rbend=" << rBend << " sbend=" << sBend
        << " tau=" << tau
        << " dx=" << dx << " dt=" << converter.getPhysDeltaT()
        << " uLat=" << converter.getCharLatticeVelocity()
        << " Re=" << 2.0*U_MEAN * 2.0*RADIUS / NU_WATER / 2.0
        << " iTmax=" << iTmax
        << std::endl;

  util::Timer<T> timer(iTmax, geometry.getStatistics().getNvoxel());
  timer.start();
  int iT = 0;
  for (; iT < iTmax; ++iT) {
    lattice.collideAndStream();
    if (iT % iTcheck == 0) {
      converge.takeValue(lattice.getStatistics().getAverageEnergy(), false);
      if (converge.hasConverged()) {
        clout << "converged at iT=" << iT << std::endl;
        break;
      }
    }
  }
  timer.stop();

  lattice.setProcessingContext(ProcessingContext::Evaluation);
  SuperLatticePhysVelocity3D<T,DESCRIPTOR> uPhysRaw(lattice, converter);
  AnalyticalFfromSuperF3D<T> uInterpRaw(uPhysRaw, true);
  SuperLatticeDensity3D<T,DESCRIPTOR> rhoLat(lattice);
  AnalyticalFfromSuperF3D<T> rhoInterp(rhoLat, true);
  // DENSITY-WEIGHTED velocity: steady LB flow down a long pipe at this
  // diffusive-scaled dt carries an O(10%) density drop (measured +9.8%
  // volume-flux rise, predicted 11.5% from Delta-p/c_s^2), so the
  // conserved and physically meaningful field is rho*u/rho_ref — the
  // incompressible-limit velocity. rho_ref is the MEASURED mean density
  // on the mother reference section: the inlet imposes the analytic
  // VELOCITY at the local (elevated) density, so mass flux everywhere
  // carries the inlet density, and normalising by anything else leaves a
  // global scale error (measured +14% with rho_ref = 1, the outlet's).
  T rhoRef = T(0);
  {
    const int nr = 24, nth = 16;
    int cnt = 0;
    for (int ir = 0; ir < nr; ++ir) {
      const T r = (T(ir) + 0.5) / T(nr) * (RADIUS - 0.51*dx);
      for (int it = 0; it < nth; ++it) {
        const T th = 2.0 * M_PI * T(it) / T(nth);
        T pt[3] = {L.x0, r*std::cos(th), r*std::sin(th)};
        T rho[1] = {T(1)};
        rhoInterp(rho, pt);
        rhoRef += rho[0];
        ++cnt;
      }
    }
    rhoRef /= T(cnt);
  }
  auto uInterp = [&](T out[3], const T in[3]) {
    T u[3] = {T(0), T(0), T(0)};
    T rho[1] = {T(1)};
    T pt[3] = {in[0], in[1], in[2]};
    uInterpRaw(u, pt);
    rhoInterp(rho, pt);
    const T w = rho[0] / rhoRef;
    out[0] = u[0]*w; out[1] = u[1]*w; out[2] = u[2]*w;
  };

  // (a) full field on the shared grid: every cell centre within a cell of
  // the bore, physical units. The scalar app's --ufield loader keys on
  // round(2 x / dx), so exact coordinates are what matter, not layout.
  {
    std::ofstream csv(outdir + "ufield.csv");
    csv.precision(10);
    csv << "# x, y, z, ux, uy, uz  (phys); dx=" << dx << "\n";
    const T r2max = (RADIUS + 1.5*dx) * (RADIUS + 1.5*dx);
    const int ni = int(L.xMax / dx) + 2;
    const int nj = int((L.yMax - L.yMin) / dx) + 2;
    const int nk = int(2.0*(RADIUS + L.pad) / dx) + 2;
    // INTEGER-grid sample points: OpenLB places lattice nodes at integer
    // multiples of dx from the box origin (verified by the geometry
    // census: material extrema sit exactly on i*dx). The first dump used
    // the half-shifted grid and every scalar-side lookup missed - the
    // field silently read zero and the scalar crawled (lag +51 t2).
    for (int i = 0; i < ni; ++i) {
      for (int j = 0; j < nj; ++j) {
        for (int k = 0; k < nk; ++k) {
          const T p[3] = {T(i) * dx,
                          L.yMin + T(j) * dx,
                          -(RADIUS + L.pad) + T(k) * dx};
          if (f.radial2(p) > r2max) { continue; }
          const T sPath = f.pathOf(p);
          if (sPath < T(0) || sPath > L.pathLen + 2.0*dx) { continue; }
          T u[3] = {T(0), T(0), T(0)};
          uInterp(u, p);
          csv << p[0] << ", " << p[1] << ", " << p[2] << ", "
              << u[0] << ", " << u[1] << ", " << u[2] << "\n";
        }
      }
    }
  }

  // (b) diameter profiles: straight mother (at x0) and mid-daughter,
  // u projected on the local tangent, stations in the bend plane.
  {
    std::ofstream csv(outdir + "profiles.csv");
    csv.precision(12);
    csv << "# station, y_over_a, u_tangential_phys\n";
    const int nSample = 201;
    struct Station { const char* name; Vector<T,3> centre, normal, tang; };
    const Vector<T,3> nD(-f.d2[1], f.d2[0], T(0));
    Station st[2] = {
      {"mother", Vector<T,3>(L.x0, T(0), T(0)),
       Vector<T,3>(T(0), T(1), T(0)), Vector<T,3>(T(1), T(0), T(0))},
      {"daughter", f.B1 + f.d2 * (0.5 * L.daughterLen), nD, f.d2},
    };
    for (const auto& S : st) {
      for (int k = 0; k < nSample; ++k) {
        const T yLoc = -RADIUS + (2.0 * RADIUS) * T(k) / T(nSample - 1);
        const T yEval = util::max(-RADIUS + 0.51 * dx,
                                  util::min(RADIUS - 0.51 * dx, yLoc));
        const Vector<T,3> p = S.centre + S.normal * yEval;
        T pt[3] = {p[0], p[1], p[2]};
        T u[3] = {T(0), T(0), T(0)};
        uInterp(u, pt);
        csv << S.name << ", " << yEval / RADIUS << ", "
            << u[0]*S.tang[0] + u[1]*S.tang[1] + u[2]*S.tang[2] << "\n";
      }
    }
  }

  // (c) fluxes through three cross-sections (disc quadrature, 24 x 16),
  // for the G2 balance gate; plus the P2 minimum tangential velocity on
  // the bend centreline region.
  T flux[3] = {T(0), T(0), T(0)};
  {
    struct Sect { Vector<T,3> centre, t; };
    const Vector<T,3> nD(-f.d2[1], f.d2[0], T(0));
    Sect sec[3] = {
      {Vector<T,3>(L.x0, T(0), T(0)), Vector<T,3>(T(1), T(0), T(0))},
      {f.B1 + f.d2 * (0.1 * L.daughterLen), f.d2},
      {f.B1 + f.d2 * (0.8 * L.daughterLen), f.d2},
    };
    for (int sIdx = 0; sIdx < 3; ++sIdx) {
      const Vector<T,3>& tv = sec[sIdx].t;
      const Vector<T,3> e1 = (sIdx == 0)
        ? Vector<T,3>(T(0), T(1), T(0)) : nD;
      const Vector<T,3> e2(T(0), T(0), T(1));
      const int nr = 24, nth = 16;
      for (int ir = 0; ir < nr; ++ir) {
        const T r = (T(ir) + 0.5) / T(nr) * (RADIUS - 0.51*dx);
        for (int it = 0; it < nth; ++it) {
          const T th = 2.0 * M_PI * T(it) / T(nth);
          const Vector<T,3> p = sec[sIdx].centre
            + e1 * (r * std::cos(th)) + e2 * (r * std::sin(th));
          T pt[3] = {p[0], p[1], p[2]};
          T u[3] = {T(0), T(0), T(0)};
          uInterp(u, pt);
          const T uT = u[0]*tv[0] + u[1]*tv[1] + u[2]*tv[2];
          flux[sIdx] += uT * r
            * ((RADIUS - 0.51*dx) / T(nr)) * (2.0 * M_PI / T(nth));
        }
      }
    }
  }
  T uMinBend = T(1e9);
  if (f.angle > T(0)) {
    const int nS = 32;
    for (int k = 0; k <= nS; ++k) {
      const T phi = f.angle * T(k) / T(nS);
      const Vector<T,3> p = f.C
        + Vector<T,3>(f.R*std::sin(phi), -f.R*std::cos(phi), T(0));
      T pt[3] = {p[0], p[1], p[2]};
      T u[3] = {T(0), T(0), T(0)};
      uInterp(u, pt);
      const T uT = u[0]*std::cos(phi) + u[1]*std::sin(phi);
      uMinBend = util::min(uMinBend, uT);
    }
  }

  clout << "betaflow-gates"
        << " rho_ref_mother=" << rhoRef
        << " steps_run=" << iT
        << " flux_mother=" << flux[0]
        << " flux_daughter_near=" << flux[1]
        << " flux_daughter_far=" << flux[2]
        << " u_min_bend_centreline=" << (f.angle > T(0) ? uMinBend : T(0))
        << std::endl;

  clout << "betaflow-done" << std::endl;
  return 0;
}
