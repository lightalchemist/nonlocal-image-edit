#define CATCH_CONFIG_MAIN 
#include "catch.hpp"
#include "filter.hpp"
#include "utils.hpp"

#include <algorithm>

const double tol = 1e-20;

TEST_CASE("OpenCV and Eigen conversions", "[utils]")
{

    SECTION("Size match") {
        cv::Mat m = cv::Mat::ones(2, 5, nle::OPENCV_MAT_TYPE);
        nle::Vec v1 = nle::opencv2Eigen<nle::DType>(m);
        CHECK(v1.size() == 10);

    }

    SECTION("All ones") {
        cv::Mat m = cv::Mat::ones(2, 5, nle::OPENCV_MAT_TYPE);
        nle::Vec v1 = nle::opencv2Eigen<nle::DType>(m);
        nle::Vec v2 = nle::Vec::Ones(10);
        CHECK(v1.isApprox(v2, tol));
    }

    SECTION("Sequence") {
        cv::Mat m = (cv::Mat_<nle::DType>(3,3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
        nle::Vec v1 = nle::opencv2Eigen<nle::DType>(m);
        nle::Vec v2 = Eigen::ArrayXd::LinSpaced(9, 1, 9);
        CHECK(v1.size() == 9);
        CHECK(v1.isApprox(v2, tol));
    }

    SECTION("OpenCV to Eigen and back") {
        cv::Mat m1 = (cv::Mat_<nle::DType>(3,3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
        nle::Vec v1 = nle::opencv2Eigen<nle::DType>(m1);
        cv::Mat m2 = nle::eigen2opencv(v1, m1.rows, m1.cols);
        CHECK(std::equal(m1.begin<nle::DType>(), m1.end<nle::DType>(), m2.begin<nle::DType>()));
    }
}

TEST_CASE("Eigen Decomposition", "[numerics]")
{
    // nle::Mat R = nle::Mat::Random(5, 5);
    nle::Mat R(3, 3);
    R << 1, 1, 0, 1, 1, 1, 0, 1, 1;
    // R = ((R + R.transpose()) / 2).eval();

    SECTION("Eigen decomposition is exact") {
        nle::Mat U;
        nle::Vec D;
        std::tie(U, D) = nle::eigenDecomposition(R, tol);

        std::cout << "D: " << std::endl;
        std::cout << D << std::endl;

        std::cout << "U: " << std::endl;
        std::cout << U << std::endl;

        CHECK((U * D.asDiagonal() * U.transpose()).isApprox(R, tol));

        std::cout << "R:" << std::endl;
        std::cout << R << std::endl;
        std::cout << "Reconstruction" << std::endl;
        std::cout << (U * D.diagonal() * U.transpose()) << std::endl;
    }
}


TEST_CASE("Sinkhorn", "[numerics]")
{
    nle::Mat I = nle::Mat::Identity(2, 2);
    nle::Vec eigvals = nle::Vec::Ones(2);
    nle::Mat Wa, Wab;
    std::tie(Wa, Wab) = nle::sinkhornKnopp(I, eigvals, 50);

    SECTION("Wa is symmetric") {
        CHECK(Wa.isApprox(Wa.transpose()));
    }

    SECTION("Row sum to 1") {
        nle::Mat Wblock(Wa.rows(), Wa.cols() + Wab.cols());
        Wblock << Wa, Wab;

        nle::Vec ones = nle::Vec(Wa.rows());
        CHECK(Wblock.rowwise().sum().isApprox(ones, tol));

        // std::cout << Wblock.colwise().sum().rows() << std::endl;
        // std::cout << "Wblock" << std::endl;
        // std::cout << Wblock << std::endl;
        // std::cout << "Wa" << std::endl;
        // std::cout << Wa << std::endl;
        // std::cout << "Wab" << std::endl;
        // std::cout << Wab << std::endl;
    }

    SECTION("Col sum to 1") {
        nle::Mat Wblock(Wa.rows() + Wab.cols(), Wa.cols());
        Wblock << Wa, Wab.transpose();

        nle::Vec ones = nle::Vec(Wa.cols());
        CHECK(Wblock.colwise().sum().isApprox(ones, tol));
    }


    nle::Mat R = nle::Mat::Random(3, 3);
    R.array() = (R.array() + 1) / 2;
    SECTION("Balanced random matrix") {
        nle::Mat U;
        nle::Vec D;
        std::tie(U, D) = nle::eigenDecomposition(R, tol);
        nle::Mat Wa, Wab;
        std::tie(Wa, Wab) = nle::sinkhornKnopp(U, D, 50);

        SECTION("Wa is symmetric") {
            CHECK(Wa.isApprox(Wa.transpose(), tol));
        }

        SECTION("Col sum to 1") {
            nle::Mat Wblock(Wa.rows() + Wab.cols(), Wa.cols());
            Wblock << Wa, Wab.transpose();

            nle::Vec ones = nle::Vec::Ones(Wa.cols());
            CHECK(Wblock.colwise().sum().isApprox(ones, tol));
        }

        SECTION("Row sum to 1") {
            nle::Mat Wblock(Wa.rows(), Wa.cols() + Wab.cols());
            Wblock << Wa, Wab;
            nle::Vec ones = nle::Vec::Ones(Wa.rows());
            std::cout << "Ones: " << std::endl;
            std::cout << ones << std::endl;
            CHECK(Wblock.rowwise().sum().isApprox(ones, tol));

            std::cout << "Wblock: " << std::endl;
            std::cout << Wblock << std::endl;
            std::cout << "Wblock rowwise sum: " << std::endl;
            std::cout << Wblock.rowwise().sum() << std::endl;
        }

        std::cout << R << std::endl;

        // nle::Mat Wblock(Wa.rows(), Wa.cols() + Wab.cols());
        // Wblock << Wa, Wab;
        // std::cout << "Wblock" << std::endl;
        // std::cout << Wblock << std::endl;
        //
        // std::cout << Wblock.rowwise().sum() << std::endl;
    }

}
