/*  junctionFlow3d — the solved flow through the pre-registered Y-junction.
 *
 *  D3Q19 steady flow: developed Poiseuille fed at the mother inlet, equal
 *  pressure at BOTH daughter outlets (the natural symmetric-split BC),
 *  Bouzidi walls on the union bore (mother cylinder + two back-extended
 *  Murray daughters — junctionGeometry.h is the shared definition). Warm
 *  start from the region-wise Poiseuille field. Machinery inherited from
 *  bentFlow3d, including its three measured lessons: the inlet Dirichlet
 *  push after initialize, the compressibility density head (the dumped
 *  field is rho*u normalised by the measured inlet-section density), and
 *  the integer-grid dump convention with the scalar-side guard.
 *
 *  Gates printed for the runner (pre-registration G2/G3/P2):
 *    flux_mother, flux_daughter_plus, flux_daughter_minus  (quadrature)
 *    u_min_daughter_centreline  (P2: no recirculation past the carina)
 *
 *  CLI: --resolution (cells per mother RADIUS, default 12), --tau (0.8),
 *  --angle (half-angle deg, 30), --sjunction (m, 350e-6), --horizon
 *  (layout match with the scalar, 6.5), --maxt (s, 3.0), --outdir.
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

#include "junctionGeometry.h"
using junction::RADIUS;
using junction::RADIUS_D;
using junction::U_MEAN;
using junction::U_MEAN_D;
using junction::NU_WATER;

class WarmU : public AnalyticalF3D<T,T> {
  const junction::JunctionFrame _f;
  T _dLen;
public:
  WarmU(const junction::JunctionFrame& f, T dLen)
    : AnalyticalF3D<T,T>(3), _f(f), _dLen(dLen) {}
  bool operator()(T out[], const T in[]) override {
    _f.warmU(in, out, _dLen);
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
  OstreamManager clout(std::cout, "junctionFlow3d");
  initialize(&argc, &argv);

  const int  res     = int(argOpt(argc, argv, "--resolution", 12));
  const T    tau     = argOpt(argc, argv, "--tau", 0.8);
  const T    angleDeg = argOpt(argc, argv, "--angle", 30.0);
  const T    sJun    = argOpt(argc, argv, "--sjunction", 350e-6);
  const T    horizon = argOpt(argc, argv, "--horizon", 6.5);
  const T    maxt    = argOpt(argc, argv, "--maxt", 3.0);
  const std::string wall = argStr(argc, argv, "--wall", "bouzidi");
  const std::string outdir = argStr(argc, argv, "--outdir", "./tmp/");
  singleton::directories().setOutputDir(outdir);

  const junction::JunctionLayout L(res, horizon, angleDeg, sJun);
  const junction::JunctionFrame& f = L.f;
  const T dx = L.dx;

  Vector<T,3> extent(L.xMax, 2.0*L.yAbsMax, 2.0*(RADIUS + L.pad));
  Vector<T,3> origin(T(0), -L.yAbsMax, -(RADIUS + L.pad));
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
  IndicatorCylinder3D<T> mother(Vector<T,3>(2.0*dx, T(0), T(0)),
                                Vector<T,3>(f.xj, T(0), T(0)), RADIUS);
  IndicatorCylinder3D<T> daughterP(f.B - f.dPlus * RADIUS,
                                   L.pEndPlus, RADIUS_D);
  IndicatorCylinder3D<T> daughterM(f.B - f.dMinus * RADIUS,
                                   L.pEndMinus, RADIUS_D);
  geometry.rename(2, 1, mother);
  geometry.rename(2, 1, daughterP);
  geometry.rename(2, 1, daughterM);
  // Inlet disc at the mother start; separate pressure outlets at each
  // daughter end (unconditional disc renames — the conditional variant
  // matched nothing, the bentFlow3d lesson).
  IndicatorCylinder3D<T> inflow(Vector<T,3>(T(0), T(0), T(0)),
                                Vector<T,3>(1.6*dx, T(0), T(0)),
                                RADIUS - 0.1*dx);
  geometry.rename(2, 3, inflow);
  IndicatorCylinder3D<T> outP(L.pEndPlus + f.dPlus * (0.4*dx),
                              L.pEndPlus + f.dPlus * (2.0*dx),
                              RADIUS_D - 0.1*dx);
  geometry.rename(2, 4, outP);
  IndicatorCylinder3D<T> outM(L.pEndMinus + f.dMinus * (0.4*dx),
                              L.pEndMinus + f.dMinus * (2.0*dx),
                              RADIUS_D - 0.1*dx);
  geometry.rename(2, 5, outM);
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
  if (wall == "bb") {
    boundary::set<boundary::BounceBack>(lattice, geometry, 2);
  } else {
    using olb::IndicatorF3D;
    std::shared_ptr<IndicatorF3D<T>> surf(
      new IndicatorCylinder3D<T>(Vector<T,3>(2.0*dx - 4.0*dx, T(0), T(0)),
                                 Vector<T,3>(f.xj, T(0), T(0)), RADIUS));
    surf = surf + std::shared_ptr<IndicatorF3D<T>>(
      new IndicatorCylinder3D<T>(f.B - f.dPlus * RADIUS,
                                 L.pEndPlus + f.dPlus * (4.0*dx),
                                 RADIUS_D));
    surf = surf + std::shared_ptr<IndicatorF3D<T>>(
      new IndicatorCylinder3D<T>(f.B - f.dMinus * RADIUS,
                                 L.pEndMinus + f.dMinus * (4.0*dx),
                                 RADIUS_D));
    setBouzidiBoundary<T, DESCRIPTOR, BouzidiPostProcessor>(
      lattice, geometry, 2, *surf);
  }
  boundary::set<boundary::InterpolatedVelocity>(lattice, geometry, 3);
  boundary::set<boundary::InterpolatedPressure>(lattice, geometry, 4);
  boundary::set<boundary::InterpolatedPressure>(lattice, geometry, 5);

  WarmU uWarm(f, L.daughterLen);
  momenta::setVelocity(lattice, geometry.getMaterialIndicator({1, 3}),
                       uWarm);
  lattice.setParameter<descriptors::OMEGA>(
    converter.getLatticeRelaxationFrequency());
  lattice.initialize();
  momenta::setVelocity(lattice, geometry.getMaterialIndicator({3}), uWarm);
  lattice.setProcessingContext<
    Array<momenta::FixedVelocityMomentumGeneric::VELOCITY>>(
    ProcessingContext::Simulation);

  const int iTmax = int(std::ceil(maxt / converter.getPhysDeltaT()));
  const int iTcheck = util::max(1, iTmax / 200);
  util::ValueTracer<T> converge(iTcheck, 1e-11);

  clout << "betaflow-provenance"
        << " angle_deg=" << angleDeg
        << " sjunction=" << sJun
        << " radius_d=" << RADIUS_D
        << " tau=" << tau
        << " dx=" << dx << " dt=" << converter.getPhysDeltaT()
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

  // (a) field dump on the lattice's integer grid, cells within 1.5 dx of
  // either bore.
  {
    std::ofstream csv(outdir + "ufield.csv");
    csv.precision(10);
    csv << "# x, y, z, ux, uy, uz  (phys); dx=" << dx << "\n";
    const int ni = int(L.xMax / dx) + 2;
    const int nj = int(2.0*L.yAbsMax / dx) + 2;
    const int nk = int(2.0*(RADIUS + L.pad) / dx) + 2;
    for (int i = 0; i < ni; ++i) {
      for (int j = 0; j < nj; ++j) {
        for (int k = 0; k < nk; ++k) {
          const T p[3] = {T(i) * dx,
                          -L.yAbsMax + T(j) * dx,
                          -(RADIUS + L.pad) + T(k) * dx};
          bool near = false;
          if (p[0] <= f.xj + dx
              && f.r2Mother(p) <= (RADIUS+1.5*dx)*(RADIUS+1.5*dx)) {
            near = true;
          }
          if (!near) {
            for (int sgn : {+1, -1}) {
              T s, r2;
              f.daughterCoords(p, sgn, s, r2);
              if (s >= -RADIUS && s <= L.daughterLen + 2.0*dx
                  && r2 <= (RADIUS_D+1.5*dx)*(RADIUS_D+1.5*dx)) {
                near = true;
                break;
              }
            }
          }
          if (!near) { continue; }
          T u[3] = {T(0), T(0), T(0)};
          uInterp(u, p);
          csv << p[0] << ", " << p[1] << ", " << p[2] << ", "
              << u[0] << ", " << u[1] << ", " << u[2] << "\n";
        }
      }
    }
  }

  // (b) profiles: mother at x0 (in-plane diameter), each daughter at
  // mid-length (in-plane diameter, tangential projection).
  {
    std::ofstream csv(outdir + "profiles.csv");
    csv.precision(12);
    csv << "# station, y_over_a_local, u_tangential_phys\n";
    struct Station {
      const char* name; Vector<T,3> centre, normal, tang; T aLoc;
    };
    const Vector<T,3> nP(-f.dPlus[1], f.dPlus[0], T(0));
    const Vector<T,3> nM(-f.dMinus[1], f.dMinus[0], T(0));
    Station st[3] = {
      {"mother", Vector<T,3>(L.x0, T(0), T(0)),
       Vector<T,3>(T(0), T(1), T(0)), Vector<T,3>(T(1), T(0), T(0)),
       RADIUS},
      {"daughter_plus", f.B + f.dPlus * (0.5 * L.daughterLen), nP,
       f.dPlus, RADIUS_D},
      {"daughter_minus", f.B + f.dMinus * (0.5 * L.daughterLen), nM,
       f.dMinus, RADIUS_D},
    };
    const int nSample = 201;
    for (const auto& S : st) {
      for (int k = 0; k < nSample; ++k) {
        const T yLoc = -S.aLoc + (2.0 * S.aLoc) * T(k) / T(nSample - 1);
        const T yEval = util::max(-S.aLoc + 0.51 * dx,
                                  util::min(S.aLoc - 0.51 * dx, yLoc));
        const Vector<T,3> p = S.centre + S.normal * yEval;
        T pt[3] = {p[0], p[1], p[2]};
        T u[3] = {T(0), T(0), T(0)};
        uInterp(u, pt);
        csv << S.name << ", " << yEval / S.aLoc << ", "
            << u[0]*S.tang[0] + u[1]*S.tang[1] + u[2]*S.tang[2] << "\n";
      }
    }
  }

  // (c) fluxes (G2, G3) and the P2 minimum along both daughter
  // centrelines from the carina to two mother radii downstream.
  T flux[3] = {T(0), T(0), T(0)};
  {
    struct Sect { Vector<T,3> centre, t, e1; T aLoc; };
    const Vector<T,3> nP(-f.dPlus[1], f.dPlus[0], T(0));
    const Vector<T,3> nM(-f.dMinus[1], f.dMinus[0], T(0));
    Sect sec[3] = {
      {Vector<T,3>(L.x0, T(0), T(0)), Vector<T,3>(T(1), T(0), T(0)),
       Vector<T,3>(T(0), T(1), T(0)), RADIUS},
      {f.B + f.dPlus * (0.5 * L.daughterLen), f.dPlus, nP, RADIUS_D},
      {f.B + f.dMinus * (0.5 * L.daughterLen), f.dMinus, nM, RADIUS_D},
    };
    for (int sIdx = 0; sIdx < 3; ++sIdx) {
      const Vector<T,3> e2(T(0), T(0), T(1));
      const int nr = 24, nth = 16;
      const T aEff = sec[sIdx].aLoc - 0.51*dx;
      for (int ir = 0; ir < nr; ++ir) {
        const T r = (T(ir) + 0.5) / T(nr) * aEff;
        for (int it = 0; it < nth; ++it) {
          const T th = 2.0 * M_PI * T(it) / T(nth);
          const Vector<T,3> p = sec[sIdx].centre
            + sec[sIdx].e1 * (r * std::cos(th)) + e2 * (r * std::sin(th));
          T pt[3] = {p[0], p[1], p[2]};
          T u[3] = {T(0), T(0), T(0)};
          uInterp(u, pt);
          const T uT = u[0]*sec[sIdx].t[0] + u[1]*sec[sIdx].t[1]
                     + u[2]*sec[sIdx].t[2];
          flux[sIdx] += uT * r * (aEff / T(nr)) * (2.0 * M_PI / T(nth));
        }
      }
    }
  }
  T uMin = T(1e9);
  for (int sgn : {+1, -1}) {
    const Vector<T,3>& d = (sgn > 0) ? f.dPlus : f.dMinus;
    const int nS = 32;
    for (int k = 0; k <= nS; ++k) {
      const T s = 2.0 * RADIUS * T(k) / T(nS);
      const Vector<T,3> p = f.B + d * s;
      T pt[3] = {p[0], p[1], p[2]};
      T u[3] = {T(0), T(0), T(0)};
      uInterp(u, pt);
      uMin = util::min(uMin, u[0]*d[0] + u[1]*d[1] + u[2]*d[2]);
    }
  }

  clout << "betaflow-gates"
        << " rho_ref_mother=" << rhoRef
        << " steps_run=" << iT
        << " flux_mother=" << flux[0]
        << " flux_daughter_plus=" << flux[1]
        << " flux_daughter_minus=" << flux[2]
        << " u_min_daughter_centreline=" << uMin
        << std::endl;

  clout << "betaflow-done" << std::endl;
  return 0;
}
