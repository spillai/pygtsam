// Author(s): Sudeep Pillai (spillai@csail.mit.edu)
// License: MIT

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/stl_iterator.hpp>

// pybot_eigen_types
#include <pygtsam/pybot_eigen_types.hpp>

// gtsam
#include <gtsam/base/types.h>
#include <gtsam/base/FastVector.h>

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/geometry/StereoPoint2.h>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/BearingRangeFactor.h>
#include <gtsam/slam/RangeFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/SmartProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include <gtsam/nonlinear/NonlinearISAM.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearEquality.h>

#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace py = boost::python;
namespace gt = gtsam;
namespace NM = gt::noiseModel;

// ====================================================================
// GTSAM Python Module
// ====================================================================

namespace bot { namespace python { 

#define DEFINE_EXTRACT_VALUE_FROM_VALUES(FUNCTION_NAME, TYPE)                    \
  std::map<gt::Symbol, TYPE> FUNCTION_NAME(const gt::Values& values) { \
    gt::Values::ConstFiltered<TYPE> poses = values.filter<TYPE>(); \
    std::map<gt::Symbol, TYPE> result;                             \
    BOOST_FOREACH(const gt::Values::ConstFiltered<TYPE>::KeyValuePair& key_value, poses) { \
      result[gt::Symbol(key_value.key)] = key_value.value;              \
    }                                                                   \
    return result;                                                      \
  }                                                                     \

DEFINE_EXTRACT_VALUE_FROM_VALUES(extractPose2, gt::Pose2);
DEFINE_EXTRACT_VALUE_FROM_VALUES(extractPose3, gt::Pose3);
DEFINE_EXTRACT_VALUE_FROM_VALUES(extractPoint3, gt::Point3);

// /// Extract all Pose2 values into vector of matrices
// std::map<gt::Symbol, gt::Pose2> extractPose2(const gt::Values& values) {
//   gt::Values::ConstFiltered<gt::Pose2> poses = values.filter<gt::Pose2>();
//   std::map<gt::Symbol, gt::Pose2> result;
//   BOOST_FOREACH(const gt::Values::ConstFiltered<gt::Pose2>::KeyValuePair& key_value, poses) {
//     result[gt::Symbol(key_value.key)] = key_value.value;
//   }
//   return result;
// }

// /// Extract all Pose3 values into vector of matrices
// std::map<gt::Symbol, gt::Pose3> extractPose3(const gt::Values& values) {
//   gt::Values::ConstFiltered<gt::Pose3> poses = values.filter<gt::Pose3>();
//   std::map<gt::Symbol, gt::Pose3> result;
//   BOOST_FOREACH(const gt::Values::ConstFiltered<gt::Pose3>::KeyValuePair& key_value, poses) {
//     result[gt::Symbol(key_value.key)] = key_value.value;
//   }
//   return result;
// }


// /// Extract all Point3 values into vector of matrices
// std::map<gt::Symbol, gt::Point3> extractPoint3(const gt::Values& values) {
//   gt::Values::ConstFiltered<gt::Point3> poses = values.filter<gt::Point3>();
//   std::map<gt::Symbol, gt::Point3> result;
//   BOOST_FOREACH(const gt::Values::ConstFiltered<gt::Point3>::KeyValuePair& key_value, poses) {
//     result[gt::Symbol(key_value.key)] = key_value.value;
//   }
//   return result;
// }

gt::ISAM2Result gtISAM2update_remove_list(gt::ISAM2& isam, const gt::NonlinearFactorGraph& newFactors, gt::Values& newTheta, const py::object& remove_list) {
  const std::vector<size_t> removeFactorIndices = std::vector<size_t>(py::stl_input_iterator<size_t>( remove_list ),
                                                                      py::stl_input_iterator<size_t>());
  return isam.update(newFactors, newTheta, removeFactorIndices);
}

void gt_ISAM2_printFactors(gt::ISAM2& isam) {
  const gt::NonlinearFactorGraph& nl = isam.getFactorsUnsafe();
  std::cout << "ISAM2FactorGraph" << std::endl;
  nl.print();
}

py::list extractKeys(const gt::Values& values) {
  py::list d;
  BOOST_FOREACH(const gt::Values::ConstKeyValuePair& key_val, values)
  {
    d.append(key_val.key);
  }
  return d;  
}

py::dict extractKeyValues(const gt::Values& values) {
  py::dict d;
  BOOST_FOREACH(const gt::Values::ConstKeyValuePair& key_val, values)
  {
    d[key_val.key] = key_val.value;
  }
  return d;  
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(gtPose3compose,
                                       gt::Pose3::compose, 1, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(gtPose3transform_to,
                                       gt::Pose3::transform_to, 1, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(gtPose3transform_from,
                                       gt::Pose3::transform_from, 1, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(gtPose2compose,
                                       gt::Pose2::compose, 1, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(gtISAM2update,
                                       gt::ISAM2::update, 0, 7)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(gtSimpleCameraproject, 
                                       gt::SimpleCamera::project, 1, 4)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(gtSymbolprint, gt::Symbol::print, 0, 1)

// template <typename DERIVED>
// DERIVED asPose3(const gt::DerivedValue<DERIVED>& a) { 
//   return static_cast<const DERIVED&>(a);
// }


// template <typename T>
// struct pointee<SmartPointer<T> >
// {
//   typedef T type;
// };

// } // namespace python
// } // namespace boost


// namespace detail {
// // @brief Construct Source from Target.
// template <typename Source,
//           typename Target>
// Source construct_helper(Target& target)
// {
//   // Lookup the construct function via ADL.  The second argument is
//   // used to:
//   // - Encode the type to allow for template's to deduce the desired
//   //   return type without explicitly requiring all construct functions
//   //   to be a template.
//   // - Disambiguate ADL when a matching convert function is declared
//   //   in both Source and Target's enclosing namespace.  It should
//   //   prefer Target's enclosing namespace.
//   return construct(target, static_cast<boost::type<Source>*>(NULL));
// }
// } // namespace detail

// /// @brief Enable implicit conversions between Source and Target types
// ///        within Boost.Python.
// ///
// ///        The conversion of Source to Target should be valid with
// ///        `Target t(s);` where `s` is of type `Source`.
// ///
// ///        The conversion of Target to Source will use a helper `construct`
// ///        function that is expected to be looked up via ADL.
// ///
// ///        `Source construct(Target&, boost::type<Source>*);`
// template <typename Source,
//           typename Target>
// struct two_way_converter
// {
//   two_way_converter()
//   {
//     // Enable implicit source to target conversion.
//     boost::python::implicitly_convertible<Source, Target>();

//     // Enable target to source conversion, that will use the convert
//     // helper.
//     boost::python::converter::registry::push_back(
//         &two_way_converter::convertible,
//         &two_way_converter::construct,
//         boost::python::type_id<Source>()
//                                                   );
//   }

//   /// @brief Check if PyObject contains the Source pointee type.
//   static void* convertible(PyObject* object)
//   {
//     // The object is convertible from Target to Source, if:
//     // - object contains Target.
//     // - object contains Source's pointee.  The pointee type must be
//     //   used, as this is the converter for Source.  Extracting Source
//     //   would cause Boost.Python to invoke this function, resulting
//     //   infinite recursion.
//     typedef typename boost::python::pointee<Source>::type pointee;
//     return boost::python::extract<Target>(object).check() &&
//         boost::python::extract<pointee>(object).check()
//                 ? object
//         : NULL;
//   }

//   /// @brief Convert PyObject to Source type.
//   static void construct(
//       PyObject* object,
//       boost::python::converter::rvalue_from_python_stage1_data* data)
//   {
//     namespace python = boost::python;

//     // Obtain a handle to the memory block that the converter has allocated
//     // for the C++ type.
//     typedef python::converter::rvalue_from_python_storage<Source>
//         storage_type;
//     void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

//     // Extract the target.
//     Target target = boost::python::extract<Target>(object);

//     // Allocate the C++ type into the converter's memory block, and assign
//     // its handle to the converter's convertible variable.  The C++ type
//     // will be copy constructed from the return of construct function.
//     data->convertible = new (storage) Source(
//         detail::construct_helper<Source>(target));
//   }
// };

// /// @brief Construct SmartPointer<Derived> from a SmartPointer<Base>.
// template <typename Derived>
// Derived construct(const SmartPointer<Base>& base, boost::type<Derived>*)
// {
//   // Assumable, this would need to do more for a true smart pointer.
//   // Otherwise, two unrelated sets of smart pointers are managing the
//   // same instance.
//   return Derived(base.get());
// }


#define DEFINE_DERIVED_VALUE(NAME, TYPE)                \
  py::class_<gt::DerivedValue<TYPE>,                    \
             py::bases<gt::Value>,                      \
             boost::noncopyable >(NAME, py::no_init)    \
  // .def("__repr__", &gt::DerivedValue<TYPE>::print)       \
  .def("printf", &gt::DerivedValue<TYPE>::print)         \
  // .def("get", &gt::DerivedValue<TYPE>::operator())   \
  ;                                                     \

BOOST_PYTHON_MODULE(pygtsam)
{
  // Main types export
  bot::python::init_and_export_eigen_converters();
  py::scope scope = py::scope();

  // --------------------------------------------------------------------
  // Optionals
  python_optional<gt::Matrix>();
  python_optional<gt::Point3>();
  python_optional<gt::Pose3>();
  
  // --------------------------------------------------------------------

  const gt::FastVector<gt::Key>& (gt::Factor::*gt_Factor_const_keys)() const = &gt::Factor::keys;
  
  // Factor
  py::class_<gt::Factor>
      ("Factor", py::no_init)
      .def("keys", gt_Factor_const_keys,
           py::return_value_policy<py::copy_const_reference>())
      ;

  // NonlinearFactor
  py::class_<gt::NonlinearFactor, py::bases<gt::Factor>,
             boost::shared_ptr<gt::NonlinearFactor>, boost::noncopyable>
      ("NonlinearFactor", py::no_init)
      // .def("error", py::pure_virtual(&gt::NonlinearFactor::error))
      // .def("dim", py::pure_virtual(&gt::NonlinearFactor::dim))
      // .def("linearize", py::pure_virtual(&gt::NonlinearFactor::linearize))
      ;
  
  // GaussianFactor
  py::class_<gt::GaussianFactor, py::bases<gt::Factor>,
             boost::shared_ptr<gt::GaussianFactor>, boost::noncopyable>
      ("GaussianFactor", py::no_init)
      ;

  
  // --------------------------------------------------------------------
  // NoiseModelFactors

  py::class_<gt::NoiseModelFactor,
             py::bases<gt::NonlinearFactor>, 
             boost::shared_ptr<gt::NoiseModelFactor>,
             boost::noncopyable>
      ("NoiseModelFactor", py::no_init)
      // .def("get_noiseModel", &gt::NoiseModelFactor::get_noiseModel)
      ;
  
  py::class_<gt::NoiseModelFactor1<gt::Pose2>,
             py::bases<gt::NoiseModelFactor>, 
             boost::shared_ptr<gt::NoiseModelFactor1<gt::Pose2> >,
             boost::noncopyable > 
      ("NoiseModelFactor1Pose2", py::no_init)
      ;

  py::class_<gt::NoiseModelFactor1<gt::Pose3>,
             py::bases<gt::NoiseModelFactor>, 
             boost::shared_ptr<gt::NoiseModelFactor1<gt::Pose3> >,
             boost::noncopyable > 
      ("NoiseModelFactor1Pose3", py::no_init)
      ;

  py::class_<gt::NoiseModelFactor1<gt::Point2>,
             py::bases<gt::NoiseModelFactor>, 
             boost::shared_ptr<gt::NoiseModelFactor1<gt::Point2> >,
             boost::noncopyable > 
      ("NoiseModelFactor1Point2", py::no_init)
      ;

  py::class_<gt::NoiseModelFactor2<gt::Pose2, gt::Pose2>,
             py::bases<gt::NoiseModelFactor>, 
             boost::shared_ptr<gt::NoiseModelFactor2<gt::Pose2, gt::Pose2> >,
             boost::noncopyable > 
      ("NoiseModelFactor2Pose2", py::no_init)
      ;

  py::class_<gt::NoiseModelFactor2<gt::Pose3, gt::Pose3>,
             py::bases<gt::NoiseModelFactor>,
             boost::shared_ptr<gt::NoiseModelFactor2<gt::Pose3, gt::Pose3> >,
             boost::noncopyable > 
      ("NoiseModelFactor2Pose3", py::no_init)
      ;

  py::class_<gt::NoiseModelFactor2<gt::Pose2, gt::Point2>,
             py::bases<gt::NoiseModelFactor>, 
             boost::shared_ptr<gt::NoiseModelFactor2<gt::Pose2, gt::Point2> >,
             boost::noncopyable > 
      ("NoiseModelFactor2Pose2Point2", py::no_init)
      ;

  py::class_<gt::NoiseModelFactor2<gt::Pose3, gt::Point3>,
             py::bases<gt::NoiseModelFactor>, 
             boost::shared_ptr<gt::NoiseModelFactor2<gt::Pose3, gt::Point3> >,
             boost::noncopyable > 
      ("NoiseModelFactor2Pose3Point3", py::no_init)
      ;

  py::class_<gt::PriorFactor<gt::Pose2>,
             py::bases<gt::NoiseModelFactor1<gt::Pose2> >, 
             boost::shared_ptr<gt::PriorFactor<gt::Pose2> >,
             boost::noncopyable> 
      ("PriorFactorPose2",
       py::init<gt::Key, gt::Pose2, gt::SharedNoiseModel>())
      ;

  py::class_<gt::PriorFactor<gt::Pose3>,
             py::bases<gt::NoiseModelFactor1<gt::Pose3> >, 
             boost::shared_ptr<gt::PriorFactor<gt::Pose3> >,
             boost::noncopyable> 
      ("PriorFactorPose3",
       py::init<gt::Key, gt::Pose3, gt::SharedNoiseModel>())
      ;

  py::class_<gt::PriorFactor<gt::Point2>,
             py::bases<gt::NoiseModelFactor1<gt::Point2> >, 
             boost::shared_ptr<gt::PriorFactor<gt::Point2> >,
             boost::noncopyable> 
      ("PriorFactorPoint2",
       py::init<gt::Key, gt::Point2, gt::SharedNoiseModel>())
      ;

  py::class_<gt::BetweenFactor<gt::Pose2>,
             py::bases<gt::NoiseModelFactor2<gt::Pose2, gt::Pose2> >, 
             boost::shared_ptr<gt::BetweenFactor<gt::Pose2> >,
             boost::noncopyable> 
      ("BetweenFactorPose2",
       py::init<gt::Key, gt::Key, gt::Pose2, gt::SharedNoiseModel>())
      ;

  py::class_<gt::BetweenFactor<gt::Pose3>,
             py::bases<gt::NoiseModelFactor2<gt::Pose3, gt::Pose3> >, 
             boost::shared_ptr<gt::BetweenFactor<gt::Pose3> >,
             boost::noncopyable> 
      ("BetweenFactorPose3",
       py::init<gt::Key, gt::Key, gt::Pose3, gt::SharedNoiseModel>())
      ;

  // Should be templated
  py::class_<gt::NonlinearEquality<gt::Pose3>,
             py::bases<gt::NoiseModelFactor1<gt::Pose3> >, 
             boost::shared_ptr<gt::NonlinearEquality<gt::Pose3> >,
             boost::noncopyable> 
      ("NonlinearEqualityPose3",
       py::init<gt::Key, gt::Pose3, py::optional<bool> >())
      ;
  
  py::class_<gt::BearingRangeFactor<gt::Pose2, gt::Point2>,
             py::bases<gt::NoiseModelFactor2<gt::Pose2, gt::Point2> >, 
             boost::shared_ptr<gt::BearingRangeFactor<gt::Pose2, gt::Point2> >,
             boost::noncopyable> 
      ("BearingRangeFactorPose2Point2",
       py::init<gt::Key, gt::Key, gt::Rot2, double, gt::SharedNoiseModel>())
      ;

  py::class_<gt::RangeFactor<gt::Pose2, gt::Point2>,
             py::bases<gt::NoiseModelFactor2<gt::Pose2, gt::Point2> >, 
             boost::shared_ptr<gt::RangeFactor<gt::Pose2, gt::Point2> >,
             boost::noncopyable> 
      ("RangeFactorPose2Point2",
       py::init<gt::Key, gt::Key, double, gt::SharedNoiseModel>())
      ;

  py::class_<gt::GenericProjectionFactor<gt::Pose3, gt::Point3, gt::Cal3_S2>,
             py::bases<gt::NoiseModelFactor2<gt::Pose3, gt::Point3> >, 
             boost::shared_ptr<gt::GenericProjectionFactor<gt::Pose3, gt::Point3, gt::Cal3_S2> >,
             boost::noncopyable> 
      ("GenericProjectionFactorPose3Point3Cal3_S2",
       py::init<const gt::Point2&, const gt::SharedNoiseModel&,
       gt::Key, gt::Key, const boost::shared_ptr<gt::Cal3_S2>&, 
       py::optional<boost::optional<gt::Pose3> > > (
           (py::arg("measured"), py::arg("model"),
            py::arg("poseKey"), py::arg("landmarkKey"),
            py::arg("K"), py::arg("body_P_sensor"))))
      ;
  
  py::class_<gt::GenericStereoFactor<gt::Pose3, gt::Point3>,
             py::bases<gt::NoiseModelFactor2<gt::Pose3, gt::Point3> >, 
             boost::shared_ptr<gt::GenericStereoFactor<gt::Pose3, gt::Point3> >,
             boost::noncopyable> 
      ("GenericStereoFactor3D",
       py::init<const gt::StereoPoint2&, const gt::SharedNoiseModel&,
       gt::Key, gt::Key, const boost::shared_ptr<gt::Cal3_S2Stereo>&, 
       py::optional<boost::optional<gt::Pose3> > > (
           (py::arg("measured"), py::arg("model"),
            py::arg("poseKey"), py::arg("landmarkKey"),
            py::arg("K"), py::arg("body_P_sensor"))))
      ;

  // SmartFactorBasePose3Cal3_S2D6
  typedef gt::SmartFactorBase<gt::Pose3, gt::Cal3_S2, 6> SmartFactorBasePose3Cal3_S2D6;
  
  void (SmartFactorBasePose3Cal3_S2D6::*SmartFactorBasePose3Cal3_S2D6_add_single)
      (const gt::Point2& measured_i, const gt::Key& poseKey_i,
       const gt::SharedNoiseModel& noise_i) = &SmartFactorBasePose3Cal3_S2D6::add;

  void (SmartFactorBasePose3Cal3_S2D6::*SmartFactorBasePose3Cal3_S2D6_add_multiple)
      (std::vector<gt::Point2>& measurements, std::vector<gt::Key>& poseKeys,
       std::vector<gt::SharedNoiseModel>& noises) = &SmartFactorBasePose3Cal3_S2D6::add;

  // Naming this class SmartFactor seels
  // SmartFactorBasePose3Cal3_S2D6
  py::class_<SmartFactorBasePose3Cal3_S2D6, py::bases<gt::NonlinearFactor>,
      boost::shared_ptr<SmartFactorBasePose3Cal3_S2D6 >,
      boost::noncopyable>
      ("SmartFactorBasePose3Cal3_S2D6", py::no_init)
      .def("add_single", SmartFactorBasePose3Cal3_S2D6_add_single)
      .def("add_multiple", SmartFactorBasePose3Cal3_S2D6_add_multiple)
      .def("measured", &SmartFactorBasePose3Cal3_S2D6::measured,
           py::return_value_policy<py::copy_const_reference>())
      .def("noise", &SmartFactorBasePose3Cal3_S2D6::noise,
           py::return_value_policy<py::copy_const_reference>())
      .def("error", &SmartFactorBasePose3Cal3_S2D6::error)
      ;
  
  // LinearizationMode
  py::enum_<gt::LinearizationMode>("LinearizationMode")
      .value("HESSIAN", gt::LinearizationMode::HESSIAN)
      .value("JACOBIAN_SVD", gt::LinearizationMode::JACOBIAN_SVD)
      .value("JACOBIAN_Q", gt::LinearizationMode::JACOBIAN_Q)
      ;

  // SmartProjectionFactorPose3Point3Cal3_S2D6
  typedef gt::SmartProjectionFactor<gt::Pose3, gt::Point3, gt::Cal3_S2, 6> SmartProjectionFactorPose3Point3Cal3_S2D6;

  boost::optional<gt::Point3> (SmartProjectionFactorPose3Point3Cal3_S2D6::*SmartProjectionFactorPose3Point3Cal3_S2D6_point_return)
      () const = &SmartProjectionFactorPose3Point3Cal3_S2D6::point;
  boost::optional<gt::Point3> (SmartProjectionFactorPose3Point3Cal3_S2D6::*SmartProjectionFactorPose3Point3Cal3_S2D6_point_compute)
      (const gt::Values& values) const  = &SmartProjectionFactorPose3Point3Cal3_S2D6::point;

  py::class_<SmartProjectionFactorPose3Point3Cal3_S2D6,
             py::bases< SmartFactorBasePose3Cal3_S2D6 >,
             boost::shared_ptr<SmartProjectionFactorPose3Point3Cal3_S2D6>,
             boost::noncopyable>
      ("SmartProjectionFactorPose3Point3Cal3_S2D6", py::no_init)
      .def("point_return", SmartProjectionFactorPose3Point3Cal3_S2D6_point_return)
      .def("point_compute", SmartProjectionFactorPose3Point3Cal3_S2D6_point_compute)
      .def("isDegenerate", &SmartProjectionFactorPose3Point3Cal3_S2D6::isDegenerate)
      .def("isPointBehindCamera", &SmartProjectionFactorPose3Point3Cal3_S2D6::isPointBehindCamera)
      ;
  
  // SmartProjectionPoseFactorPose3Point3Cal3_S2
  typedef gt::SmartProjectionPoseFactor<gt::Pose3, gt::Point3, gt::Cal3_S2> SmartProjectionPoseFactorPose3Point3Cal3_S2;

  void (SmartProjectionPoseFactorPose3Point3Cal3_S2::*SmartProjectionPoseFactorPose3Point3Cal3_S2_add_single)
      (const gt::Point2 measured_i, const gt::Key poseKey_i,
       const gt::SharedNoiseModel noise_i,
       const boost::shared_ptr<gt::Cal3_S2> K_i) = &SmartProjectionPoseFactorPose3Point3Cal3_S2::add;

  // void (SmartProjectionPoseFactorPose3Point3Cal3_S2::*SmartProjectionPoseFactorPose3Point3Cal3_S2_add_multiple_with_different_noises)
  //     (std::vector<gt::Point2> measurements, std::vector<gt::Key> poseKeys,
  //      std::vector<gt::SharedNoiseModel> noises,
  //      std::vector<boost::shared_ptr<gt::Cal3_S2> > Ks) = &SmartProjectionPoseFactorPose3Point3Cal3_S2::add;

  // void (SmartProjectionPoseFactorPose3Point3Cal3_S2::*SmartProjectionPoseFactorPose3Point3Cal3_S2_add_multiple_with_shared_noise)
  //     (std::vector<gt::Point2> measurements, std::vector<gt::Key> poseKeys,
  //      const gt::SharedNoiseModel noise, 
  //      std::vector<boost::shared_ptr<gt::Cal3_S2> > Ks) = &SmartProjectionPoseFactorPose3Point3Cal3_S2::add;

  py::class_<gt::SmartProjectionPoseFactor<gt::Pose3, gt::Point3, gt::Cal3_S2>,
             py::bases< SmartProjectionFactorPose3Point3Cal3_S2D6 >,
             boost::shared_ptr<gt::SmartProjectionPoseFactor<gt::Pose3, gt::Point3, gt::Cal3_S2> >,
             boost::noncopyable>
      ("SmartFactor", 
       py::init<py::optional<double, double, bool, bool,
       boost::optional<gt::Pose3>, gt::LinearizationMode, double, double> >(
           (py::arg("rankTol")=1, py::arg("linThreshold")=-1, py::arg("manageDegeneracy")=false,
            py::arg("enableEPI")=false, py::arg("body_P_sensor"),
            py::arg("linearizeTo")=gt::LinearizationMode::HESSIAN, py::arg("landmarkDistanceThreshold")=1e10,
            py::arg("dynamicOutlierRejectionTreshold")=-1)))
      .def("add_single", SmartProjectionPoseFactorPose3Point3Cal3_S2_add_single)
      // .def("add_multiple", SmartProjectionPoseFactorPose3Point3Cal3_S2_add_multiple_with_different_noises)
      // .def("add_multiple_with_shared_noise", SmartProjectionPoseFactorPose3Point3Cal3_S2_add_multiple_with_shared_noise)
      ;

             
  // --------------------------------------------------------------------
  // NoiseModels
  
  // Noise models
  // py::scope noiseModel
  {
    // boost::shared_ptr<gt::noiseModel::Base>
    py::class_<gt::noiseModel::Base, boost::noncopyable >("Base", py::no_init)
        .def("print", py::pure_virtual(&gt::noiseModel::Base::print))
        ;

    // implicitly_convertible< Derived::Ptr, Base::Ptr >();
    
    // The usage of smart pointers (e.g. boost::shared_ptr<T>) is another common way
    // to give away ownership of objects in C++. These kinds of smart pointer are
    // automatically handled if you declare their existence when declaring the class
    // to boost::python. This is done by including the holding type as a template
    // parameter to class_<>, like in the following example:
    
    py::class_<gt::noiseModel::Gaussian, py::bases<gt::noiseModel::Base>,
               boost::shared_ptr<gt::noiseModel::Gaussian>, boost::noncopyable>("Gaussian", py::no_init)
        .def("SqrtInformation", &gt::noiseModel::Gaussian::SqrtInformation)
        .staticmethod("SqrtInformation")
        .def("Information", &gt::noiseModel::Gaussian::Information)
        .staticmethod("Information")
        .def("Covariance", &gt::noiseModel::Gaussian::Covariance)
        .staticmethod("Covariance")
        ;

    // py::class_<gt::noiseModel::Diagonal, py::bases<gt::noiseModel::Gaussian> >("Diagonal", py::no_init)
    py::class_<gt::noiseModel::Diagonal, py::bases<gt::noiseModel::Gaussian>,
               boost::shared_ptr<gt::noiseModel::Diagonal>, boost::noncopyable>("Diagonal", py::no_init)
        .def("Sigmas", &gt::noiseModel::Diagonal::Sigmas, 
             (py::arg("sigmas"), py::arg("smart")=true))
        .staticmethod("Sigmas")
        .def("Variances", &gt::noiseModel::Diagonal::Variances, 
             (py::arg("variances"), py::arg("smart")=true))
        .staticmethod("Variances")
        .def("Precisions", &gt::noiseModel::Diagonal::Precisions, 
             (py::arg("precisions"), py::arg("smart")=true))
        .staticmethod("Precisions")
        ;
    // boost::python::register_ptr_to_python<boost::shared_ptr<gt::noiseModel::Diagonal> >();
    
    
    py::class_<gt::noiseModel::Constrained, py::bases<gt::noiseModel::Diagonal>,
               boost::shared_ptr<gt::noiseModel::Constrained>, boost::noncopyable>("Constrained", py::no_init)
        // .def("MixedSigmas", &gt::noiseModel::Constrained::MixedSigmas)
        // .staticmethod("MixedSigmas")
        // .def("MixedVariances", &gt::noiseModel::Constrained::MixedVariances)
        // .staticmethod("MixedVariances")
        // .def("MixedPrecisions", &gt::noiseModel::Constrained::MixedPrecisions)
        // .staticmethod("MixedPrecisions")
        // .def("All", &gt::noiseModel::Constrained::All)
        // .staticmethod("All")
        ;

    py::class_<gt::noiseModel::Isotropic, py::bases<gt::noiseModel::Diagonal>,
               boost::shared_ptr<gt::noiseModel::Isotropic>, boost::noncopyable>("Isotropic", py::no_init)
        .def("Sigma", &gt::noiseModel::Isotropic::Sigma,
             (py::arg("dim"), py::arg("sigma"), py::arg("smart")=true))
        .staticmethod("Sigma")
        .def("Variance", &gt::noiseModel::Isotropic::Variance,
             (py::arg("dim"), py::arg("variance"), py::arg("smart")=true))
        .staticmethod("Variance")
        .def("Precision", &gt::noiseModel::Isotropic::Precision,
             (py::arg("dim"), py::arg("precision"), py::arg("smart")=true))
        .staticmethod("Precision")
        ;

    py::class_<gt::noiseModel::Unit, py::bases<gt::noiseModel::Isotropic>,
               boost::shared_ptr<gt::noiseModel::Unit>, boost::noncopyable>("Unit", py::no_init)
        .def("Create", &gt::noiseModel::Unit::Create)
        ;
  }
  
  // --------------------------------------------------------------------
  // Keys, Values, DerivedValue

  // KeyFormatter
  py::class_<gt::KeyFormatter>("KeyFormatter")
      ;
  
  // Value
  py::class_<gt::Value, boost::noncopyable>("Value", py::no_init)
      // .def("deallocate_", py::pure_virtual(&gt::Value::deallocate_))
      // .def("clone", py::pure_virtual(&gt::Value::clone))
      // .def("clone_", py::pure_virtual(&gt::Value::clone_),
      //      py::return_value_policy<py::manage_new_object>())
      // .def("equals_", py::pure_virtual(&gt::Value::equals_))
      // .def("print", py::pure_virtual(&gt::Value::print))
      // .def("dim", py::pure_virtual(&gt::Value::dim))
      // .def("retract_", py::pure_virtual(&gt::Value::retract_),
      //           py::return_value_policy<py::manage_new_object>())
      // .def("localCoordinates_", py::pure_virtual(&gt::Value::localCoordinates_))
      ;

  // DerivedValue
  DEFINE_DERIVED_VALUE("DerivedValuePoint2", gt::Point2);
  DEFINE_DERIVED_VALUE("DerivedValuePoint3", gt::Point3);
  DEFINE_DERIVED_VALUE("DerivedValueStereoPoint2", gt::StereoPoint2);
  DEFINE_DERIVED_VALUE("DerivedValueRot2", gt::Rot2);
  DEFINE_DERIVED_VALUE("DerivedValueRot3", gt::Rot3);
  DEFINE_DERIVED_VALUE("DerivedValuePose2", gt::Pose2);
  DEFINE_DERIVED_VALUE("DerivedValuePose3", gt::Pose3);
  DEFINE_DERIVED_VALUE("DerivedValueCal3_S2", gt::Cal3_S2);
  DEFINE_DERIVED_VALUE("DerivedValueSimpleCamera", gt::SimpleCamera);
  
  // py::class_<gt::DerivedValue<gt::Pose2>, py::bases<gt::Value> >("DerivedValue", py::no_init)
  //     // .def("deallocate_", py::pure_virtual(&gt::DerivedValue<gt::Pose2>::deallocate_))
  //     // .def("clone", py::pure_virtual(&gt::DerivedValue<gt::Pose2>::clone))
  //     // .def("clone_", py::pure_virtual(&gt::DerivedValue<gt::Pose2>::clone_),
  //     //      py::return_value_policy<py::manage_new_object>())
  //     // .def("equals_", py::pure_virtual(&gt::DerivedValue<gt::Pose2>::equals_))
  //     // .def("print", py::pure_virtual(&gt::DerivedValue<gt::Pose2>::print))
  //     // .def("dim", py::pure_virtual(&gt::DerivedValue<gt::Pose2>::dim))
  //     // .def("retract_", py::pure_virtual(&gt::DerivedValue<gt::Pose2>::retract_),
  //     //           py::return_value_policy<py::manage_new_object>())
  //     // .def("localCoordinates_", py::pure_virtual(&gt::DerivedValue<gt::Pose2>::localCoordinates_))


  // FastLists
  custom_list_to_list<gt::Key>();
  py::class_< gt::FastVector<gt::Key> >("KeyVector")
      .def(py::vector_indexing_suite< gt::FastVector<gt::Key> >())
      ;

  py::class_< gt::FastList<gt::Key> >("KeyList")
      // .def("__call__", &gt::FastList<gt::Key>::operator)
      // .def("__iter__", py::range(&gt::FastList<gt::Key>::begin,
      //                            &gt::FastList<gt::Key>::end))
      // .def(py::vector_indexing_suite< gt::FastList<gt::Key> >())
      ;
  
  // Values
  void (gt::Values::*gt_Values_insert1)(gt::Key j, const gt::Value& val) = &gt::Values::insert;
  void (gt::Values::*gt_Values_insert2)(const gt::Values& values) = &gt::Values::insert;
  void (gt::Values::*gt_Values_update)(gt::Key j, const gt::Value& val) = &gt::Values::update;
  // typedef const gt::Value& (gt::Values::*gt_Values_at1)(gt::Key j) const;
  const gt::Value& (gt::Values::*gt_Values_at1)(gt::Key j) const = &gt::Values::at;

  // /// Const forward iterator, with value type ConstKeyValuePair
  // typedef boost::transform_iterator<
  //   boost::function1<ConstKeyValuePair, const ConstKeyValuePtrPair&>, KeyValueMap::const_iterator> const_iterator;

  // gt::const_iterator (gt::Values::*gt_Values_const_begin)() const = &gt::Values::begin;
  py::class_<gt::Values>("Values", py::init<>())
      // .def("__iter__", py::range(&gt::Values::begin, &gt::Values::end))
      .def("keys", &gt::Values::keys)
      .def("insert", gt_Values_insert1)
      .def("insert", gt_Values_insert2)
      .def("clear", &gt::Values::clear)
      .def("update", gt_Values_update)
      .def("at", gt_Values_at1,
           py::return_value_policy<py::copy_const_reference>())

      // .def("iteritems", py::range(&gt::Values::begin, &gt::Values::end))
      .def("atPoint2", &gt::Values::at<gt::Point2>,
           py::return_value_policy<py::copy_const_reference>())
      .def("atRot2", &gt::Values::at<gt::Rot2>,
           py::return_value_policy<py::copy_const_reference>())
      .def("atPose2", &gt::Values::at<gt::Pose2>,
           py::return_value_policy<py::copy_const_reference>())
      .def("atPoint3", &gt::Values::at<gt::Point3>,
           py::return_value_policy<py::copy_const_reference>())
      .def("atRot3", &gt::Values::at<gt::Rot3>,
           py::return_value_policy<py::copy_const_reference>())
      .def("atPose3", &gt::Values::at<gt::Pose3>,
           py::return_value_policy<py::copy_const_reference>())
      // .def("atVector3", &gt::Values::at<gt::Vector3>,
      //      py::return_value_policy<py::copy_const_reference>())


      
      .def("printf", &gt::Values::print,
           (py::arg("s")="", py::arg("formatter")=gt::DefaultKeyFormatter))
      ;

  // KeyFormatter
  py::enum_<gt::Marginals::Factorization>("Factorization")
      .value("QR", gt::Marginals::Factorization::QR)
      .value("CHOLESKY", gt::Marginals::Factorization::CHOLESKY)
      ;
  
  // Marginals
  py::class_<gt::Marginals>("Marginals",
                            py::init<gt::NonlinearFactorGraph,
                            gt::Values,
                            py::optional<gt::Marginals::Factorization> >())
      .def("marginalCovariance", &gt::Marginals::marginalCovariance)
      .def("printf", &gt::Marginals::print,
           (py::arg("s")="", py::arg("formatter")=gt::DefaultKeyFormatter))
      ;

  // --------------------------------------------------------------------
  // Point2, Point3, Rot2, Rot3, Pose2, Pose3
  
  // Point2
  py::class_<gt::Point2,
             py::bases<gt::DerivedValue<gt::Point2> > >
      ("Point2", py::init<>())
      .def(py::init<double, double>())
      .def(py::init<gt::Vector>())
      .def("vector", &gt::Point2::vector)
      ;
  expose_template_type<gt::Point2>();
  expose_template_type<std::vector<gt::Point2> >();

  // Point3
  py::class_<gt::Point3,
             py::bases<gt::DerivedValue<gt::Point3> > >
      ("Point3", py::init<>())
      .def(py::init<double, double, double>())
      .def(py::init<gt::Vector>())
      .def("vector", &gt::Point3::vector)
      ;

  // StereoPoint2
  py::class_<gt::StereoPoint2,
             py::bases<gt::DerivedValue<gt::StereoPoint2> > >
      ("StereoPoint2", py::init<>())
      .def(py::init<double, double, double>
           (py::args("uL", "uR", "v")))
      .def("identity", &gt::StereoPoint2::identity)
      .staticmethod("identity")
      .def("print", &gt::StereoPoint2::print)
      .def("equals", &gt::StereoPoint2::equals)
      .def("inverse", &gt::StereoPoint2::inverse)
      .def("compose", &gt::StereoPoint2::compose)
      // .def("operator+", &gt::StereoPoint2::operator+)
      // .def("operator-", &gt::StereoPoint2::operator-)
      .def("uL", &gt::StereoPoint2::uL)
      .def("uR", &gt::StereoPoint2::uR)
      .def("v", &gt::StereoPoint2::v)
      .def("vector", &gt::StereoPoint2::vector)
      .def("point2", &gt::StereoPoint2::point2)
      .def("right", &gt::StereoPoint2::right)
      ;

  // Rot2
  py::class_<gt::Rot2,
             py::bases<gt::DerivedValue<gt::Rot2> > >
      ("Rot2", py::init<>())
      .def(py::init<double>())
      .def("fromDegrees", &gt::Rot2::fromDegrees)
      .staticmethod("fromDegrees")
      .def("fromAngle", &gt::Rot2::fromAngle)
      .staticmethod("fromAngle")
      .def("fromCosSin", &gt::Rot2::fromCosSin)
      .staticmethod("fromCosSin")
      .def("relativeBearing", &gt::Rot2::relativeBearing)
      .staticmethod("relativeBearing")
      .def("atan2", &gt::Rot2::atan2)
      .staticmethod("atan2")
      .def("matrix", &gt::Rot2::matrix)
      ;


  // Rot3
  gt::Rot3 (*gtRot3rodriguez_wxwywz)
      (double wx, double wy, double wz) = &gt::Rot3::rodriguez;

  py::class_<gt::Rot3, py::bases<gt::DerivedValue<gt::Rot3> > >("Rot3", py::init<>())
      .def(py::init<double, double, double, double, double, double, double, double, double>())
      .def(py::init<gt::Point3, gt::Point3, gt::Point3>())
      .def(py::init<gt::Matrix3>())
      .def(py::init<gt::Matrix>())

      .def("Rx", &gt::Rot3::Rx)
      .staticmethod("Rx")
      .def("Ry", &gt::Rot3::Ry)
      .staticmethod("Ry")
      .def("Rz", &gt::Rot3::Rz)
      .staticmethod("Rz")

      .def("rodriguez", gtRot3rodriguez_wxwywz)
      .staticmethod("rodriguez")
      .def("matrix", &gt::Rot3::matrix)
      
      // .def("yaw", &gt::Rot3::yaw)
      // .staticmethod("yaw")
      // .def("pitch", &gt::Rot3::pitch)
      // .staticmethod("pitch")
      // .def("roll", &gt::Rot3::roll)
      // .staticmethod("roll")
      // .def("ypr", &gt::Rot3::ypr)
      // .staticmethod("ypr")
      // .def("quaternion", &gt::Rot3::quaternion)
      // .staticmethod("quaternion")
      ;

  
  // Pose2
  py::class_<gt::Pose2, py::bases<gt::DerivedValue<gt::Pose2> > >("Pose2", py::init<>())
      .def(py::init<double, double, double>())
      .def(py::init<double, gt::Point2>())
      .def(py::init<gt::Rot2, gt::Point2>())
      .def(py::init<gt::Matrix>())
      .def(py::init<gt::Vector>())
      .def("print", &gt::Pose2::print)
      .def("equals", &gt::Pose2::equals)
      .def("identity", &gt::Pose2::identity)
      .def("inverse", &gt::Pose2::inverse)
      .def("compose", &gt::Pose2::compose,
           gtPose2compose
           (py::args("p2", "H1", "H2")))
      .def("between", &gt::Pose2::between)
      .def("Dim", &gt::Pose2::Dim)
      .def("dim", &gt::Pose2::dim)
      .def("retract", &gt::Pose2::retract)
      .def("localCoordinates", &gt::Pose2::localCoordinates)
      .def("Expmap", &gt::Pose2::Expmap)
      .def("Logmap", &gt::Pose2::Logmap)
      .def("AdjointMap", &gt::Pose2::AdjointMap)
      .def("transform_to", &gt::Pose2::transform_to)
      .def("transform_from", &gt::Pose2::transform_from)
      .def("x", &gt::Pose2::x)
      .def("y", &gt::Pose2::y)
      .def("theta", &gt::Pose2::theta)
      .def("t", &gt::Pose2::t,
           py::return_value_policy<py::reference_existing_object>())
      .def("r", &gt::Pose2::r,
           py::return_value_policy<py::reference_existing_object>())
      .def("translation", &gt::Pose2::translation,
           py::return_value_policy<py::reference_existing_object>())
      .def("rotation", &gt::Pose2::rotation,
           py::return_value_policy<py::reference_existing_object>())
      .def("matrix", &gt::Pose2::matrix)
      ;

  // Pose3  
  py::class_<gt::Pose3, py::bases<gt::DerivedValue<gt::Pose3> > >
      ("Pose3", py::init<>())
      .def(py::init<gt::Rot3, gt::Point3>())
      .def(py::init<gt::Pose2>())
      .def(py::init<gt::Matrix>())
      .def("print", &gt::Pose3::print)
      .def("equals", &gt::Pose3::equals)
      .def("identity", &gt::Pose3::identity)
      .def("inverse", &gt::Pose3::inverse)
      .def("compose", &gt::Pose3::compose,
           gtPose3compose
           (py::args("p2", "H1", "H2")))
      .def("Dim", &gt::Pose3::Dim)
      .def("dim", &gt::Pose3::dim)
      .def("retract", &gt::Pose3::retract)
      .def("localCoordinates", &gt::Pose3::localCoordinates)
      .def("Expmap", &gt::Pose3::Expmap)
      .def("Logmap", &gt::Pose3::Logmap)
      .def("AdjointMap", &gt::Pose3::AdjointMap)
      .def("transform_from", &gt::Pose3::transform_from,
           gtPose3transform_from
           (py::args("p", "Dpose", "Dpoint")))
      // .def("transform_to", &gt::Pose3::transform_to,
      //      gtPose3transform_to
      //      (py::args("p", "Dpose", "Dpoint")))
      .def("x", &gt::Pose3::x)
      .def("y", &gt::Pose3::y)
      .def("z", &gt::Pose3::z)
      // .def("t", &gt::Pose3::t,
      //      py::return_value_policy<py::reference_existing_object>())
      // .def("r", &gt::Pose3::r,
      //      py::return_value_policy<py::reference_existing_object>())
      .def("translation", &gt::Pose3::translation,
           py::return_value_policy<py::reference_existing_object>())
      .def("rotation", &gt::Pose3::rotation,
           py::return_value_policy<py::reference_existing_object>())
      .def("matrix", &gt::Pose3::matrix)
      ;

  py::def("extractPose2", &bot::python::extractPose2);
  expose_template_type<std::map<gt::Symbol, gt::Pose2> >();

  py::def("extractPose3", &bot::python::extractPose3);
  expose_template_type<std::map<gt::Symbol, gt::Pose3> >();

  py::def("extractPoint3", &bot::python::extractPoint3);
  expose_template_type<std::map<gt::Symbol, gt::Point3> >();
  
  py::def("extractKeys", &bot::python::extractKeys);
  py::def("extractKeyValues", &bot::python::extractKeyValues);

  // --------------------------------------------------------------------
  // // EssentialMatrix
  // py::class_<gt::EssentialMatrix, py::bases<gt::DerivedValue<gt::EssentialMatrix> > >
  //     ("EssentialMatrix", py::init<>())
  //     .def(py::init<gt::Rot3, gt::Unit3>())
  //     .def("print", &gt::EssentialMatrix::print)
  //     .def("equals", &gt::EssentialMatrix::equals)
  //     .def("identity", &gt::EssentialMatrix::identity)
  //     .def("inverse", &gt::EssentialMatrix::inverse)
  //     .def("compose", &gt::EssentialMatrix::compose,
  //          gtEssentialMatrixcompose
  //          (py::args("p2", "H1", "H2")))
  //     .def("Dim", &gt::EssentialMatrix::Dim)
  //     .def("dim", &gt::EssentialMatrix::dim)
  //     // .def("retract", &gt::EssentialMatrix::retract)
  //     // .def("localCoordinates", &gt::EssentialMatrix::localCoordinates)
  //     .def("translation", &gt::EssentialMatrix::translation,
  //          py::return_value_policy<py::reference_existing_object>())
  //     .def("direction", &gt::EssentialMatrix::rotation,
  //          py::return_value_policy<py::reference_existing_object>())
  //     .def("matrix", &gt::EssentialMatrix::matrix)
  //     ;
  
  py::class_<gt::Cal3_S2, py::bases<gt::DerivedValue<gt::Cal3_S2> > >
      ("Cal3_S2", py::init<>())
      .def(py::init<double, double, double, double, double>(
          py::args("fx", "fy", "s", "u0", "v0")))
      .def(py::init<gt::Vector>())
      .def(py::init<double, int, int>(py::args("fov", "w", "h")))
      
      .def("print", &gt::Cal3_S2::print)
      .def("equals", &gt::Cal3_S2::equals)

      .def("Dim", &gt::Cal3_S2::Dim)
      .def("dim", &gt::Cal3_S2::dim)
      .def("retract", &gt::Cal3_S2::retract)
      .def("localCoordinates", &gt::Cal3_S2::localCoordinates)
            
      .def("fx", &gt::Cal3_S2::fx)
      .def("fy", &gt::Cal3_S2::fy)
      .def("px", &gt::Cal3_S2::px)
      .def("py", &gt::Cal3_S2::py)
      .def("skew", &gt::Cal3_S2::skew)
      .def("principalPoint", &gt::Cal3_S2::principalPoint)
      .def("vector", &gt::Cal3_S2::vector)
      .def("K", &gt::Cal3_S2::K)
      .def("matrix", &gt::Cal3_S2::matrix)
      .def("matrix_inverse", &gt::Cal3_S2::matrix_inverse)
      ;

  
  py::class_<gt::Cal3_S2Stereo,
             // py::bases<gt::DerivedValue<gt::Cal3_S2Stereo> >,
             // boost::shared_ptr<gt::Cal3_S2Stereo>,
             boost::noncopyable>
      ("Cal3_S2Stereo", py::init<>())
      .def(py::init<double, double, double, double, double, double>(
          py::args("fx", "fy", "s", "u0", "v0", "b")))
      .def(py::init<gt::Vector>())
      
      .def("print", &gt::Cal3_S2Stereo::print)
      .def("equals", &gt::Cal3_S2Stereo::equals)

      .def("calibration", &gt::Cal3_S2Stereo::calibration,
           py::return_value_policy<py::reference_existing_object>())
      
      // .def("Dim", &gt::Cal3_S2Stereo::Dim)
      // .def("dim", &gt::Cal3_S2Stereo::dim)
      // .def("retract", &gt::Cal3_S2Stereo::retract)
      // .def("localCoordinates", &gt::Cal3_S2Stereo::localCoordinates)
      
      .def("fx", &gt::Cal3_S2Stereo::fx)
      .def("fy", &gt::Cal3_S2Stereo::fy)
      .def("px", &gt::Cal3_S2Stereo::px)
      .def("py", &gt::Cal3_S2Stereo::py)
      .def("skew", &gt::Cal3_S2Stereo::skew)
      .def("principalPoint", &gt::Cal3_S2Stereo::principalPoint)
      .def("baseline", &gt::Cal3_S2Stereo::baseline)
      // .def("vector", &gt::Cal3_S2Stereo::vector)
      // .def("K", &gt::Cal3_S2Stereo::K)
      .def("matrix", &gt::Cal3_S2Stereo::matrix)
      // .def("matrix_inverse", &gt::Cal3_S2Stereo::matrix_inverse)
      ;

  const gt::Pose3& (gt::SimpleCamera::*SimpleCamera_pose)
      () const = &gt::SimpleCamera::pose;

  // gt::Point2& (gt::SimpleCamera::*SimpleCamera_project)
  //     (const gt::Point3& pw, //
  //      boost::optional<gt::Matrix&> Dpose = boost::none,
  //      boost::optional<gt::Matrix&> Dpoint = boost::none,
  //      boost::optional<gt::Matrix&> Dcal = boost::none) const = &gt::SimpleCamera::project;

  py::class_<gt::SimpleCamera,
             py::bases<gt::DerivedValue<gt::SimpleCamera> > >
      ("SimpleCamera", py::init<>())
      // .def(py::init<gt::Pose3>(py::arg("pose")))
      .def(py::init<const gt::Pose3&, const gt::Cal3_S2&>(py::args("pose", "K")))
      // .def(py::init<gt::Vector>(py::arg("v")))
      // .def(py::init<gt::Vector, gt::Vector>(py::args("v", "K")))
      
      // .def("Level", &gt::SimpleCamera::Level,
      //      py::args("K", "pose2", "height"))
      // .staticmethod("Level")

      .def("Lookat", &gt::SimpleCamera::Lookat,
           (py::arg("eye"), py::arg("target"),
            py::arg("upVector"), py::arg("K")=gt::Cal3_S2()))
      .staticmethod("Lookat")

      .def("print", &gt::SimpleCamera::print)
      .def("equals", &gt::SimpleCamera::equals)
      .def("pose", SimpleCamera_pose,
           py::return_value_policy<py::copy_const_reference>())
      .def("project", &gt::SimpleCamera::project,
           gtSimpleCameraproject
           (py::args("pw", "Dpose", "Dpoint", "Dcal")))
      ;
  
  py::def("simpleCamera", &gt::simpleCamera, py::arg("P"));

  // --------------------------------------------------------------------
  // FactorGraph

  // // GaussianFactorGraph
  // py::class_<gt::FactorGraph<gt::GaussianFactor>, boost::noncopyable >
  //     ("GaussianFactorGraph", py::no_init)
  //     ;
  
  // // EliminateableFactorGraph
  // py::class_<gt::EliminateableFactorGraph<gt::GaussianFactorGraph>, boost::noncopyable>
  //     ("EliminateableGaussianFactorGraph", py::no_init)
  //     ;

  // // GaussianFactorGraph
  // py::class_<gt::GaussianFactorGraph, 
  //            py::bases<gt::FactorGraph<gt::GaussianFactor>,
  //                      gt::EliminateableFactorGraph<gt::GaussianFactorGraph> >, boost::noncopyable>
  //     ("GaussianFactorGraph", py::no_init)
  //     ;
  
  // void (gt::FactorGraph<gt::NonlinearFactor>::*add1)()             = &gt::FactorGraph<gt::NonlinearFactor>::add;
  void (gt::FactorGraph<gt::NonlinearFactor>::*FactorGraphNonlinearFactor_add2)
      (const boost::shared_ptr<gt::NonlinearFactor>& factor) = &gt::FactorGraph<gt::NonlinearFactor>::add;

  const boost::shared_ptr<gt::NonlinearFactor> (gt::FactorGraph<gt::NonlinearFactor>::*FactorGraphNonlinearFactor_at)
      (size_t i) const = &gt::FactorGraph<gt::NonlinearFactor>::at;

  py::class_<gt::FactorGraph<gt::NonlinearFactor>, boost::noncopyable>
      ("NonlinearFactorGraphBase", py::no_init)
      .def("add", FactorGraphNonlinearFactor_add2)
      .def("remove", &gt::FactorGraph<gt::NonlinearFactor>::remove)
      // .def("push_back", &gt::FactorGraph<gt::NonlinearFactor>::push_back)
      .def("__getitem__", FactorGraphNonlinearFactor_at)
      .def("resize", &gt::FactorGraph<gt::NonlinearFactor>::resize)
      .def("replace", &gt::FactorGraph<gt::NonlinearFactor>::replace)
      .def("reserve", &gt::FactorGraph<gt::NonlinearFactor>::reserve)
      .def("printf", &gt::FactorGraph<gt::NonlinearFactor>::print,
           (py::arg("s")="FactorGraph", py::arg("formatter")=gt::DefaultKeyFormatter))
      ;

  // py::class_<gt::FactorGraph<gt::NonlinearFactor>, boost::noncopyable>
  //     ("NonlinearFactorGraphBase", py::no_init)
  //     .def("add", add2)
  //     .def("printf", &gt::FactorGraph<gt::NonlinearFactor>::print,
  //          (py::arg("s")="FactorGraph", py::arg("formatter")=gt::DefaultKeyFormatter))
  //     ;

  py::class_<gt::NonlinearFactorGraph,
             py::bases<gt::FactorGraph<gt::NonlinearFactor> > >
      ("NonlinearFactorGraph")
      ;
  // py::class_<gt::GaussianFactorGraph, py::bases<gt::FactorGraph<gt::GaussianFactor> > >("GaussianFactorGraph")
  //     ;

  
  // --------------------------------------------------------------------
  // NonlienarOptimizer

  py::class_<gt::NonlinearOptimizerParams>
      ("NonlinearOptimizerParams", py::init<>())
      ;
  py::class_<gt::LevenbergMarquardtParams, py::bases<gt::NonlinearOptimizerParams> >
      ("LevenbergMarquardtParams", py::init<>())
      ;
  py::class_<gt::DoglegParams, py::bases<gt::NonlinearOptimizerParams> >
      ("DoglegParams", py::init<>())
      ;
  
  py::class_<gt::NonlinearOptimizer, boost::noncopyable>
      ("NonlinearOptimizer", py::no_init)
      .def("optimize", &gt::NonlinearOptimizer::optimize,
           py::return_value_policy<py::copy_const_reference>())
           // py::return_value_policy<py::reference_existing_object>())
      .def("optimizeSafely", &gt::NonlinearOptimizer::optimizeSafely,
           py::return_value_policy<py::copy_const_reference>())
           // py::return_value_policy<py::reference_existing_object>())
      .def("values", &gt::NonlinearOptimizer::values,
           py::return_value_policy<py::copy_const_reference>())
           // py::return_value_policy<py::reference_existing_object>())
      .def("error", &gt::NonlinearOptimizer::error)
      .def("iterations", &gt::NonlinearOptimizer::iterations)
      ;

  
  // --------------------------------------------------------------------
  // LMOptimizer

  py::class_<gt::LevenbergMarquardtOptimizer, py::bases<gt::NonlinearOptimizer> >
      ("LevenbergMarquardtOptimizer", py::init<gt::NonlinearFactorGraph, gt::Values, py::optional<gt::LevenbergMarquardtParams> >())
      ;

  // --------------------------------------------------------------------
  // DoglegOptimizer

  py::class_<gt::DoglegOptimizer, py::bases<gt::NonlinearOptimizer> >
      ("DoglegOptimizer", py::init<gt::NonlinearFactorGraph, gt::Values, py::optional<gt::DoglegParams> >())
      ;

  
  // --------------------------------------------------------------------
  // ISAM2Clique, BayesTree<ISAM2Clique>

  py::class_<gt::ISAM2Clique, boost::noncopyable>
      ("ISAM2Clique", py::no_init)
      ;

  void (gt::BayesTree<gt::ISAM2Clique>::*gt_BayesTree_ISAM2Clique_saveGraph)
      (const std::string& s,
       const gt::KeyFormatter& keyFormatter = gt::DefaultKeyFormatter) const = &gt::BayesTree<gt::ISAM2Clique>::saveGraph;

  py::class_<gt::BayesTree<gt::ISAM2Clique>, boost::noncopyable>
      ("BayesTreeISAM2Clique", py::no_init)
      .def("saveGraph", gt_BayesTree_ISAM2Clique_saveGraph,
           (py::arg("s")="", py::arg("formatter")=gt::DefaultKeyFormatter))
      ;

  // --------------------------------------------------------------------
  // ISAM2
  void (gt::ISAM2Params::*gt_ISAM2Params_print)
      (const std::string& s) const = &gt::ISAM2Params::print;

  py::class_<gt::ISAM2Params>
      ("ISAM2Params", py::init<>())
      .def(py::init<gt::ISAM2GaussNewtonParams, double, int, bool, bool, gt::ISAM2Params::Factorization, bool, const gt::KeyFormatter&>(
        py::args("optimizationParams", "relinearizeThreshold", "relinearizeSkip", "enableRelinearization", 
          "evaluateNonlinearError", "factorization", "cacheLinearizedFactors", "keyFormatter")))
      .def("printf", gt_ISAM2Params_print,
           (py::arg("s")="ISAM2Params"))
      .def("setEnableFindUnusedFactorSlots", &gt::ISAM2Params::setEnableFindUnusedFactorSlots)
      ;

  py::enum_<gt::ISAM2Params::Factorization>("ISAM2ParamsFactorization")
      .value("QR", gt::ISAM2Params::Factorization::QR)
      .value("CHOLESKY", gt::ISAM2Params::Factorization::CHOLESKY)
      ;

  py::class_<gt::ISAM2Result>
      ("ISAM2Result", py::init<>())
      .def("printf", &gt::ISAM2Result::print)
      .def("getVariablesRelinearized",
           &gt::ISAM2Result::getVariablesRelinearized)
      .def("getVariablesReeliminated",
           &gt::ISAM2Result::getVariablesReeliminated)
      .def("getCliques",
           &gt::ISAM2Result::getCliques)
      ;
  
  py::class_<gt::ISAM2GaussNewtonParams>
      ("ISAM2GaussNewtonParams", py::init<>())
      .def("print", &gt::ISAM2GaussNewtonParams::print)
      .def("getWildfireThreshold",
           &gt::ISAM2GaussNewtonParams::getWildfireThreshold)
      .def("setWildfireThreshold",
           &gt::ISAM2GaussNewtonParams::setWildfireThreshold)
      ;
  
  gt::Values (gt::ISAM2::*gt_ISAM2_calculateEstimate)
      () const = &gt::ISAM2::calculateEstimate;
  // gt::Values (gt::ISAM2::*gt_ISAM2_calculateEstimateWithKey)
  //     (gt::Key) const = &gt::ISAM2::calculateEstimate;

  py::class_<gt::ISAM2,
             py::bases<gt::BayesTree<gt::ISAM2Clique> >,
             boost::noncopyable >
      ("ISAM2", py::init<>())
      .def(py::init<gt::ISAM2Params>(py::args("ISAM2Params")))
      .def("updateRemoveFactors", gtISAM2update_remove_list)
      .def("update", &gt::ISAM2::update,
           gtISAM2update
           (py::args("newFactors", "newTheta", "removeFactorIndices",
                     "constrainedKeys", "noRelinKeys", "extraReelimKeys", "force_relinearize")))
      .def("calculateEstimate", gt_ISAM2_calculateEstimate)
      .def("marginalCovariance", &gt::ISAM2::marginalCovariance)
      // .def("calculateEstimateWithKey", gt_ISAM2_calculateEstimateWithKey)
      .def("calculateBestEstimate", &gt::ISAM2::calculateBestEstimate)
      .def("printStats", &gt::ISAM2::printStats)
      .def("printFactors", gt_ISAM2_printFactors)
      ;

  // py::class_<gt::NonlinearISAM>
  //     ("NonlinearISAM", py::init<py::optional<int, gt::GaussianFactorGraph::Eliminate> >(
  //         py::arg("reorderInterval"), py::arg("eliminationFunction")))
  //     // .def("optimize", &gt::NonlinearISAM::optimize,
  //     //      py::return_value_policy<py::reference_existing_object>())
  //     // .def("optimize", &gt::NonlinearISAM::optimize,
  //     //      py::return_value_policy<py::reference_existing_object>())
  //     .def("print", &gt::NonlinearISAM::print)
  //     .def("estimate", &gt::NonlinearISAM::estimate)
  //     .def("update", &gt::NonlinearISAM::update)
  //     .def("reorder_relinearize", &gt::NonlinearISAM::reorder_relinearize)
  //     ;

  // static boost::shared_ptr<gt::Symbol> makeSymbol(const std::string &str, size_t j) {
  //   if(str.size() != 1)
  //     throw std::runtime_error("string argument must have one character only");
  //   return boost::make_shared<gt::Symbol>(str.at(0),j);
  // }

  // // Helper function to print the symbol as "char-and-index" in python
  // std::string selfToString(const gt::Symbol & self)
  // {
  //   return (std::string)self;
  // }

  // // Helper function to convert a Symbol to int using int() cast in python
  // size_t selfToKey(const gt::Symbol & self)
  // {
  //   return self.key();
  // }

  // // Helper function to recover symbol's unsigned char as string
  // std::string chrFromSelf(const gt::Symbol & self)
  // {
  //   std::stringstream ss;
  //   ss << self.chr();
  //   return ss.str();
  // }
  
  // Symbol
  py::class_<gt::Symbol, boost::shared_ptr<gt::Symbol> >("Symbol")
      .def(py::init<>())
      .def(py::init<const gt::Symbol& >())
      // .def("__init__", py::make_constructor(makeSymbol))
      .def(py::init<unsigned char, size_t>())
      .def(py::init<gt::Key>())
      // .def("print", &gt::Symbol::print,
      //      gtSymbolprint(py::args("s")))
      .def("key", &gt::Symbol::key)
      .def("index", &gt::Symbol::index)
      .def("print", &gt::Symbol::print)
      .def("chr", &gt::Symbol::chr)
      // .def("__repr__", &selfToString)
      // .def("__int__", &selfToKey)
      ;

  py::def("symbol", &gt::symbol, (py::arg("c"), py::arg("j")));
  py::def("symbolChr", &gt::symbolChr, (py::arg("key")));
  py::def("symbolIndex", &gt::symbolIndex, (py::arg("key")));

} 
} // namespace python
} // namespace bot



  // // Vector.h
  // py::def("repeat", &gt::repeat);
  // py::def("delta", &gt::delta);
  // py::def("basis", &gt::basis);
  // py::def("zeros", &gt::zeros);
  // py::def("ones", &gt::ones);
  // py::def("zero", &gt::zero);
  // py::def("dim", &gt::dim);
  // py::def("print", &gt::print);
  // py::def("save", &gt::save);

  
  // py::def("equalto", &gt::operator==);
  
  // py::def("greaterThanOrEqual", &gt::greaterThanOrEqual);
  // py::def("equal_with_abs_tol", &gt::equal_with_abs_tol);
  // py::def("save", &gt::save);

  // Gaussian::shared_ptr Gaussian::SqrtInformation(const Matrix& R, bool smart) {
// Gaussian::shared_ptr Gaussian::Information(const Matrix& M, bool smart) {
// Gaussian::shared_ptr Gaussian::Covariance(const Matrix& covariance,


// // Point2d
//   py::class_<isam::Point2d>("Point2d")
//       .def(py::init<double, double>(py::args("x","y")))
//       // .def(py::init<Vector>())
//       ;

//   // py::class_<isam::Point2d*>("Point2dPtr")
//   //     ;

  
//   // Point3d
//   py::class_<isam::Point3d>("Point2d")
//       .def(py::init<double, double, double>(py::args("x","y","z")))
//       ;
  
//   // Pose2d
//   py::class_<isam::Pose2d>("Pose2d")
//       .def(py::init<double, double, double>(py::args("x","y","t")))
//       .add_property("x", &isam::Pose2d::x, &isam::Pose2d::set_x)
//       .add_property("y", &isam::Pose2d::y, &isam::Pose2d::set_y)
//       .add_property("z", &isam::Pose2d::t, &isam::Pose2d::set_t)
//       ;

//   // Noise
//   py::class_<isam::Noise, boost::noncopyable>("Noise")
//       .def(py::init<>())
//       ;

//   // Information
//   py::class_<isam::Information, py::bases<isam::Noise> >
//       ("Information", py::init<Eigen::MatrixXd>())
//       // .def(py::init<Eigen::MatrixXd>())
//       ;

//   // Covariance
//   py::class_<isam::Covariance, py::bases<isam::Noise> >
//       ("Covariance", py::init<Eigen::MatrixXd>())
//       // .def(py::init<Eigen::MatrixXd>())
//       ;

//   // // Element
//   // py::class_<isam::Element, boost::noncopyable>
//   //     ("Element", py::init<const char*, int>())
//   //     ;

//   // // Node
//   // py::class_<isam::Node, py::bases<isam::Element> >
//   //     ("Node", py::init<const char*, int>())
//   //     ;
  
//   // // Node
//   // py::class_<isam::Node, py::bases<isam::Element> >
//   //     ("Node", py::init<const char*, int>())
//   //     ;
  
//   // // Point2d_Node
//   // py::class_<isam::Point2d_Node, py::bases<isam::Node> >("Point2d_Node")
//   //     ;
  
//   // Pose2d_Node
//   py::class_<isam::Pose2d_Node, boost::noncopyable>("Pose2d_Node")
//       // .def("ptr", &isam::Pose2d_Node)
//       ;

//   // py::to_python_converter<Pose2d_Node*, ptr_to_python<Pose2d_Node*> >();
  
//   // Pose3d_Node
//   py::class_<isam::Pose3d_Node, boost::noncopyable>("Pose3d_Node")
//       ;


  
//   // Point2d_Factor
//   py::class_<isam::Point2d_Factor, boost::noncopyable>
//       ("Point2d_Factor", py::init<Point2d_Node*, isam::Point2d, isam::Noise>
//        (py::args("pose","prior","noise")))
//       ;
  
//   // Pose2d_Factor
//   py::class_<isam::Pose2d_Factor, boost::noncopyable>
//       ("Pose2d_Factor", py::init<Pose2d_Node*, isam::Pose2d, isam::Noise>
//        (py::args("pose","prior","noise")))
//       ;

//   // Pose2d_Pose2d_Factor
//   py::class_<isam::Pose2d_Pose2d_Factor, boost::noncopyable>
//       ("Pose2d_Pose2d_Factor",
//        py::init<isam::Pose2d_Node*, isam::Pose2d_Node*,
//        isam::Pose2d, isam::Noise>(py::args("pose1","pose2","measure","noise")))
//       // .def(py::init<isam::Pose2d_Node*, isam::Pose2d_Node*,
//       //      isam::Pose2d, isam::Noise,
//       //      isam::Anchor2d_Node*, isam::Anchor2d_Node*>
//       //      (py::arg("pose1"),py::arg("pose2"),py::arg("measure"),py::arg("noise"),
//       //       py::arg("anchor1")=0,py::arg("anchor2")=0))
//       ;
  
//   // Pose2d_Point2d_Factor
//   py::class_<isam::Pose2d_Point2d_Factor, boost::noncopyable>
//       ("Pose2d_Point2d_Factor", py::init<isam::Pose2d_Node*, isam::Point2d_Node*,
//        isam::Point2d, isam::Noise>
//        (py::args("pose","point","measure","noise")))
//       ;
  
//   // Slam
//   py::class_<isam::Slam, boost::noncopyable>("Slam", py::init<>())
//       .def("add_node", &isam::Slam::add_node)
//       .def("add_factor", &isam::Slam::add_factor)
//       .def("set_properties", &isam::Slam::set_properties)
//       .def("batch_optimization", &isam::Slam::batch_optimization)
//       .add_property("properties", &isam::Slam::properties, &isam::Slam::set_properties)
//        ;
  
  
  // // Pose3d_Factor
  // py::class_<isam::Pose2d_Factor, boost::noncopyable>
  //     ("Pose2d_Factor", py::init<Pose2d_Node*, isam::Pose2d, isam::Noise>
  //      (py::args("pose","prior","noise")))
  //     ;

  // // NonlinearFactorGraph
  // py::class_<NonlinearFactorGraph>("NonlinearFactorGraph")
  //     .def("add", &NonlinearFactorGraph::add)
  //     ;

  // // Diagonal
  // py::class_<noiseModel::Diagonal, boost::noncopyable>("Diagonal")
  //     // .def(py::init<Vector>(py::args("sigmas")))
  //     .staticmethod("Sigmas", &noiseModel::Diagonal::Sigmas)
  //     ;

  
  // // Pose2
  // py::class_<Pose2>("Pose2")
  //     .def(py::init<double, double, double>(py::args("x","y","t")))
  //     // .add_property("x", &Pose2::x, &Pose2::set_x)
  //     // .add_property("y", &Pose2::y, &Pose2::set_y)
  //     // .add_property("z", &Pose2::t, &Pose2::set_t)
  //     ;

  // Key, // SharedNoiseModel
  
  // // PriorFactorPose2
  // py::class_<PriorFactor<Pose2>, boost::noncopyable>
  //     ("PriorFactorPose2", py::init<Key, Pose2, SharedNoiseModel>
  //      (py::args("key","prior","model")))
  //     ;
