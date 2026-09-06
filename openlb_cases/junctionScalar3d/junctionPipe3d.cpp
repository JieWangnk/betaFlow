/*  junctionPipe3d — the mc_channel scalar through the Y-junction.
 *
 *  The programme's physics rung: the channel impulse response in a
 *  geometry with NO closed-form model. D3Q7 scalar with the wall and
 *  collision the G4 ladder proved out — Noble-Torczynski partially-
 *  saturated cells (solid fraction supersampled from the shared bore
 *  definition) with a TRT bulk at magic Lambda = 1/4 — advecting on the
 *  SOLVED junction flow (junctionFlow3d --ufield, with the hard guard
 *  against a silently-dead field). No prescribed-velocity mode exists
 *  here on purpose: there is no analytic junction field to prescribe.
 *
 *  FIVE receiver windows, all bulk materials (the G4 readout lesson):
 *    11  mother, path DBAR[0] — the in-run control upstream of the branch
 *    12  daughter(+), path DBAR[1]        13  daughter(+), path DBAR[2]
 *    14  daughter(−), path DBAR[1]        15  daughter(−), path DBAR[2]
 *  Windows 14/15 mirror 12/13 exactly; gate G3 compares them to round-off
 *  levels of the deterministic Eulerian solve.
 *
 *  CLI: --resolution --tau --horizon --outputs --outdir as the bent app,
 *  plus --angle (30), --sjunction (350e-6), --taueven (TRT even rate),
 *  and --ufield FILE (required).
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

// Per-cell solid fraction for the Noble-Torczynski wall.
struct SOLID_FRACTION : public descriptors::FIELD_BASE<1> { };

using MyCase = Case<
  AdvectionDiffusion,
  Lattice<double, descriptors::D3Q7<descriptors::VELOCITY, SOLID_FRACTION>>
>;
using T = MyCase::value_t;
using DESCRIPTOR = MyCase::descriptor_t_of<AdvectionDiffusion>;

#include "../junction3d/junctionGeometry.h"
using junction::RADIUS;
using junction::RADIUS_D;
using junction::U_MEAN;
using junction::DIFFUSIVITY;
using junction::CX;
using junction::DBAR;

static constexpr T CS2_D3Q7 = 0.25;

// Noble-Torczynski partially-saturated ADE dynamics with a TRT bulk —
// verbatim the scheme the G4 ladder validated (bentPipe3d.cpp carries the
// measurement history; the solid operator is exactly g_ibar - g_i on D3Q7,
// pairwise conservative; OMEGA relaxes the even sector, MAGIC places the
// odd rate that carries D, and B blends on the odd-rate tau).
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

// Solved-field loader with the dead-field guard (the G4 hand-off lesson).
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
      std::array<T,6> v{};
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

// Solid fraction: 4^3 supersampling of the shared bore, with the inlet
// cap (x < 2 dx) and both outlet caps (daughter s > daughterLen) solid.
class SolidFraction : public AnalyticalF3D<T,T> {
  const junction::JunctionFrame _f;
  T _dx, _dLen;
public:
  SolidFraction(const junction::JunctionFrame& f, T dx, T dLen)
    : AnalyticalF3D<T,T>(1), _f(f), _dx(dx), _dLen(dLen) {}
  bool operator()(T out[], const T in[]) override {
    int solid = 0;
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          const T pp[3] = {in[0] + (T(a)-1.5)/4.0*_dx,
                           in[1] + (T(b)-1.5)/4.0*_dx,
                           in[2] + (T(c)-1.5)/4.0*_dx};
          if (!_f.insideBore(pp, _dLen) || pp[0] < 2.0*_dx) { ++solid; }
        }
      }
    }
    out[0] = T(solid) / T(64);
    return true;
  }
};

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
  OstreamManager clout(std::cout, "junctionPipe3d");
  initialize(&argc, &argv);

  const int  res     = int(argOpt(argc, argv, "--resolution", 12));
  const T    tau     = argOpt(argc, argv, "--tau", 0.5048);
  const T    horizon = argOpt(argc, argv, "--horizon", 6.5);
  const int  outputs = int(argOpt(argc, argv, "--outputs", 400));
  const T    angleDeg = argOpt(argc, argv, "--angle", 30.0);
  const T    sJun    = argOpt(argc, argv, "--sjunction", 350e-6);
  const T    tauEven = argOpt(argc, argv, "--taueven", 52.583);
  const std::string ufield = argStr(argc, argv, "--ufield", "");
  const std::string outdir = argStr(argc, argv, "--outdir", "./tmp/");
  singleton::directories().setOutputDir(outdir);
  if (ufield.empty()) {
    clout << "FATAL: --ufield is required (no analytic junction field "
             "exists to prescribe)" << std::endl;
    return 1;
  }

  const junction::JunctionLayout L(res, horizon, angleDeg, sJun);
  const junction::JunctionFrame& f = L.f;
  const T dx = L.dx;
  const T dt = (tau - 0.5) * CS2_D3Q7 * dx * dx / DIFFUSIVITY;
  const int iTmax = int(std::ceil(L.tMax / dt));
  const int statIter = util::max(1, iTmax / outputs);
  const T slugHalf = 1.01 * dx;

  Vector<T,3> extent(L.xMax, 2.0*L.yAbsMax, 2.0*(RADIUS + L.pad));
  Vector<T,3> origin(T(0), -L.yAbsMax, -(RADIUS + L.pad));
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
                                Vector<T,3>(f.xj, T(0), T(0)), RADIUS);
  IndicatorCylinder3D<T> daughterP(f.B - f.dPlus * RADIUS,
                                   L.pEndPlus, RADIUS_D);
  IndicatorCylinder3D<T> daughterM(f.B - f.dMinus * RADIUS,
                                   L.pEndMinus, RADIUS_D);
  geometry.rename(2, 1, mother);
  geometry.rename(2, 1, daughterP);
  geometry.rename(2, 1, daughterM);
  // Windows as bulk materials 11..21 (readout bulk-only by construction):
  // 11 = mother control; 12..16 = daughter(+) at the P3 sweep distances
  // {600, 750, 1000, 1250, 1550} um of path; 17..21 = daughter(-) mirrors.
  static const T SWEEP[5] = {600e-6, 750e-6, 1000e-6, 1250e-6, 1550e-6};
  {
    Vector<T,3> w0, w1;
    L.windowEnds(DBAR[0], +1, w0, w1);
    IndicatorCylinder3D<T> win1(w0, w1, RADIUS);
    geometry.rename(1, 11, win1);
    int mat = 12;
    for (int sgn : {+1, -1}) {
      for (int k = 0; k < 5; ++k) {
        L.windowEnds(SWEEP[k], sgn, w0, w1);
        IndicatorCylinder3D<T> win(w0, w1, RADIUS_D);
        geometry.rename(1, mat++, win);
      }
    }
  }
  geometry.communicate();
  geometry.print();

  auto& lattice = myCase.getLattice(AdvectionDiffusion{});
  lattice.setUnitConverter<AdeUnitConverter<T,DESCRIPTOR>>(
    dx, dt, RADIUS, 2.0*U_MEAN, DIFFUSIVITY, T(1));
  const auto& converter = lattice.getUnitConverter();
  converter.print();

  auto bulkAndWalls = geometry.getMaterialIndicator({1, 2, 11, 12, 13, 14, 15,
                                                 16, 17, 18, 19, 20, 21});
  dynamics::set<NTAdeBGKdynamics>(lattice, bulkAndWalls);
  SolidFraction epsF(f, dx, L.daughterLen);
  fields::set<SOLID_FRACTION>(lattice, bulkAndWalls, epsF);

  SolvedFieldVelocity uF(converter.getConversionFactorVelocity(), dx,
                         ufield);
  {
    T probe[3] = {T(0), T(0), T(0)};
    const T at[3] = {L.x0, T(0), T(0)};
    uF(probe, at);
    const T expect = 2.0 * U_MEAN / converter.getConversionFactorVelocity();
    if (std::fabs(probe[0]) < 0.3 * expect) {
      clout << "FATAL: solved field reads " << probe[0]
            << " lattice units at the slug centre against ~" << expect
            << " — grid-convention mismatch with " << ufield << std::endl;
      return 1;
    }
    clout << "solved field loaded: " << uF.size() << " cells"
          << " (centre u_lat=" << probe[0] << ")" << std::endl;
  }
  SlugInit slugF(L.x0, 2.0 * slugHalf);
  fields::set<descriptors::VELOCITY>(lattice, bulkAndWalls, uF);
  AnalyticalConst3D<T,T> zeroRho(T(0));
  AnalyticalConst3D<T,T> zeroU(T(0), T(0), T(0));
  lattice.iniEquilibrium(geometry.getMaterialIndicator({1, 11, 12, 13, 14,
                                                        15, 16, 17, 18, 19,
                                                        20, 21}),
                         slugF, uF);
  lattice.iniEquilibrium(geometry.getMaterialIndicator({2}), zeroRho,
                         zeroU);
  lattice.setParameter<descriptors::OMEGA>(T(1) / tauEven);
  lattice.setParameter<collision::TRT::MAGIC>(
    (tauEven - 0.5) * (tau - 0.5));
  lattice.initialize();

  clout << "betaflow-provenance"
        << " angle_deg=" << angleDeg
        << " sjunction=" << sJun
        << " radius_d=" << RADIUS_D
        << " tau_requested=" << tau
        << " taueven=" << tauEven
        << " dx=" << dx << " dt=" << dt
        << " iTmax=" << iTmax << " statIter=" << statIter
        << " pathLen=" << L.pathLen << " x0=" << L.x0
        << " slugW=" << L.slugW
        << std::endl;

  clout << "checkpoint A" << std::endl;
  std::ofstream csv(outdir + "cir.csv");
  csv.precision(12);
  csv << "# t_phys, w_mother, wp_600, wp_750, wp_1000, wp_1250, wp_1550, "
         "wm_600, wm_750, wm_1000, wm_1250, wm_1550, total\n";

  clout << "checkpoint B" << std::endl;
  util::Timer<T> timer(iTmax, geometry.getStatistics().getNvoxel());
  timer.start();
  clout << "checkpoint C" << std::endl;

  for (int iT = 0; iT <= iTmax; ++iT) {
    if (iT % statIter == 0) {
      if (iT == 0) { clout << "checkpoint D" << std::endl; }
      lattice.setProcessingContext(ProcessingContext::Evaluation);
      SuperLatticeDensity3D<T,DESCRIPTOR> rho(lattice);
      if (iT == 0) { clout << "checkpoint E" << std::endl; }
      int tmp[1] = {0};
      T total[2] = {T(0), T(0)};
      T winSum[11] = {};
      for (int mat : {1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}) {
        if (iT == 0) { clout << "sum mat " << mat << std::endl; }
        // SuperSum3D writes TWO outputs: the sum and the cell count.
        // A one-element array here is a stack smash (measured: the
        // material loop's initializer list was overwritten and the
        // app died on SIGBUS; the sibling apps carried the same
        // latent overflow and survived by stack layout only).
        T part[2] = {T(0), T(0)};
        SuperSum3D<T,T> matSum(rho, geometry, mat);
        matSum(part, tmp);
        total[0] += part[0];
        if (mat >= 11) { winSum[mat - 11] = part[0]; }
      }
      csv << converter.getPhysTime(iT);
      for (int w = 0; w < 11; ++w) {
        csv << ", " << ((total[0] > T(0)) ? winSum[w]/total[0] : T(0));
      }
      csv << ", " << total[0] << "\n";
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
