#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/IterativeLinearSolvers>
// #include <Eigen/Sparse>
#include <Eigen/Eigenvalues>


double kernel(const cv::Mat& I,
              int r, int s,
              int r1, int c1, int r2, int c2,
              double spatialScale, double intensityScale) {
    
    // Ensure that I is 1 dimensional
    assert(I.rows == 1);

    // return 1.0;

    // auto yr = I.at<double>(r, 0);
    // auto ys = I.at<double>(s, 0);
    auto yr = I.at<double>(0, r);
    auto ys = I.at<double>(0, s);
    
    // Add in spatial term
    double squareSpatialDist = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2);
    double squareIntensityDist = (yr - ys) * (yr - ys);
    return std::exp(- spatialScale * squareSpatialDist - intensityScale * squareIntensityDist);
}

inline unsigned int to1DIndex(unsigned int row, unsigned int col, unsigned int ncols) {
    return row * ncols + col;
}

inline auto to2DCoords(unsigned int index, int ncols) {
    return std::make_pair(index / ncols, index % ncols);
}

double estimateVariance(const cv::Mat& I) {
    return 100.0;
}

template <typename T>
std::vector<T> convertToVec(const cv::Mat& I) {
    std::vector<T> v;
    v.reserve(I.total());
    std::copy(I.begin<T>(), I.end<T>(), std::back_inserter(v));
    
    return v;
}

auto samplePixels(unsigned int nrows, unsigned int ncols, unsigned int nRowSamples) {
    float ratio = static_cast<float>(ncols) / static_cast<float>(nrows);
    int nColSamples =  std::floor(std::fmax(ratio * nRowSamples, 1.0));

    unsigned int nPixels = nrows * ncols;
    unsigned int nSamples = nRowSamples * nColSamples;

    unsigned int rowStep = nrows / (nRowSamples);
    unsigned int colStep = ncols / (nColSamples);
    unsigned int rOffset = (rowStep + (nrows - nRowSamples * rowStep)) / 2;
    unsigned int cOffset = (colStep + (ncols - nColSamples * colStep)) / 2;
    
    std::cout << "sample pixel ratio: " << ratio << std::endl;
    std::cout << "# row samples: " << nRowSamples << " # col samples: " << nColSamples << std::endl;
    
    std::vector<int> selected;
    std::vector<int> rest;
    selected.reserve(nSamples);
    rest.reserve(nPixels - nSamples);
    
    for (auto r = 0u; r < nrows; r++) {
        for (auto c = 0u; c < ncols; c++) {
            if ((r >= rOffset && c >= cOffset) && 
                ((r - rOffset) % rowStep == 0) && ((c - cOffset) % colStep == 0)) 
            {
                selected.push_back(to1DIndex(r, c, ncols));
            }
            else {
                rest.push_back(to1DIndex(r, c, ncols));
            }
        }
    }
    
//    selected.insert(selected.end(), rest.begin(), rest.end());
    return std::make_pair(selected, rest);
}

void computeKernelWeights(const cv::Mat& I,
                          Eigen::MatrixXd& Ka,
                          Eigen::MatrixXd& Kab,
                          std::vector<int>& pixelOrder,
                          int nRowSamples = 10)
{
    auto nrows = I.rows, ncols = I.cols;
    unsigned int nPixels = I.total();
    assert(nPixels == nrows * ncols);
    
    std::vector<int> selected, rest;
    std::tie(selected, rest) = samplePixels(nrows, ncols, nRowSamples);
    int nSamples = selected.size();

    pixelOrder.reserve(nPixels);
    pixelOrder.insert(pixelOrder.end(), selected.begin(), selected.end());
    pixelOrder.insert(pixelOrder.end(), rest.begin(), rest.end());

    Ka.resize(nSamples, nSamples);
    Kab.resize(nSamples, nPixels - nSamples);

    std::cout << "nPixels: " << nPixels << std::endl;
    std::cout << "Ka size: " << Ka.rows() << " x " << Ka.cols() << std::endl;
    std::cout << "Kab size: " << Kab.rows() << " x " << Kab.cols() << std::endl;

    // TODO: Tune this
    // double variance = estimateVariance(I);
    double variance = 400;
    double gammaIntensity = 1.0 / variance;
    double gammaSpatial = 0; // 1.0 / 10;
    
    // This is row vector
    cv::Mat II = I.reshape(0, 1);
    std::cout << "II shape: " << II.size() << std::endl;
    // Compute Ka
    int r1, c1, r2, c2;
    for (auto i = 0u; i < selected.size(); ++i) {
        std::tie(r1, c1) = to2DCoords(i, ncols);
        for (auto j = 0u; j < selected.size(); ++j) {
            std::tie(r2, c2) = to2DCoords(j, ncols);
            Ka(i, j) = kernel(II, selected[i], selected[j], 
                              r1, c1, r2, c2,
                              gammaSpatial, gammaIntensity);
        }
    }
    
    for (auto i = 0u; i < Ka.rows(); i++) {
        for (auto j = 0u; j < Ka.cols(); j++) {
            if (Ka(i, j) < 0) {
                std::cout << "Ka(" << i << ", " << j << "): " << Ka(i, j) << std::endl;
                throw std::runtime_error("Ka contains negative entries");
            }
        }
    }

    std::cout << "All entries of Ka is >= 0" << std::endl;

    // Ensure that Ka is symmetric
    assert(Ka.isApprox(Ka.transpose()));

    // Check that matrix is positive definite
    // Eigen::LLT<Eigen::MatrixXd> lltOfMat(Ka.real()); // compute the Cholesky decomposition of A
    // if(lltOfMat.info() == Eigen::NumericalIssue)
    // {
    //      std::runtime_error("Possibly non positive-semi definitie matrix!");
    // }    
    

    for (auto i = 0u; i < selected.size(); i++) {
        std::tie(r1, c1) = to2DCoords(i, ncols);
        for (auto j = 0u; j < rest.size(); j++) {
            std::tie(r2, c2) = to2DCoords(j, ncols);
            Kab(i, j) = kernel(II, selected[i], rest[j],
                               r1, c1, r2, c2,
                               gammaSpatial, gammaIntensity);
        }
    }

}



auto sinkhornKnopp(const Eigen::MatrixXd& phi, const Eigen::MatrixXd& eigvals, int maxIter=20,
              double eps=0.00001)
{
   int n = phi.rows();
   // auto r = Eigen::ArrayXXd::Ones(n, 1);
   auto r = Eigen::MatrixXd::Ones(n, 1);
   std::cout << "r rows: " << r.rows() << " cols: " << r.cols() << std::endl;

   auto c = phi * (eigvals.array() * (phi.transpose() * r).array()).matrix();
   // reciprocal(c, eps);
   for (int i = 0; i < maxIter; i++) {

   }

   std::cout << "Allocating Wa Wab" << std::endl;
   Eigen::MatrixXd Wa, Wab;
   std::cout << "Allocated Wa Wab" << std::endl;


    
   return std::tie(Wa, Wab);
}

//template <typename T>
//void scaleEigenValues(const T& weights)
//{
//}

template <typename T>
auto invertDiagMatrix(const T& mat, double eps=0.00001) {
    Eigen::MatrixXd invMat = mat;
    for (int c = 0; c < mat.cols(); c++) {
        for (int r = 0; r < mat.rows(); r++) {
            if (std::abs(mat(r, c)) < eps) {
                std::cout << "Small eigenvalue: " << mat(r, c) << " at (" << r << " , " << c << ")" << std::endl;
                invMat(r, c) = 0;
            }
            else {
                invMat(r, c) = 1.0 / mat(r, c);
            }
        }
    }

    return mat.asDiagonal();
}


// template <T>
// void reciprocal(T& mat, double eps) {
//     // Iterate over each element and take reciprocal
//
// }

// auto robustReciprocal(const T& mat, double eps=0.00001) {
    
// }

// auto robustReciprocal(T& mat, double eps=0.00001) {
    
    
// }

auto nystromApproximation(const Eigen::MatrixXd& Ka, const Eigen::MatrixXd& Kab,
                          double eps=0.00001) {

    // Eigendecomposition of Ka
    Eigen::EigenSolver<Eigen::MatrixXd> es(Ka);

    // TODO: Check this. This results in a conversion to MatrixXd?
    Eigen::MatrixXd eigvals = es.eigenvalues().real();
    Eigen::MatrixXd eigvecs = es.eigenvectors().real();

    std::cout << "Finished eigendecomposition" << std::endl;

    // Approximate eigenvectors of K from the above eigenvalues and eigenvectors and Kab
    std::cout << "eigvals shape: " << eigvals.rows() << " x " << eigvals.cols() << std::endl;
    std::cout << "eigvecs shape: " << eigvecs.rows() << " x " << eigvecs.cols() << std::endl;
    
    auto invEigVals = invertDiagMatrix(eigvals, eps);
    std::cout << "invEigVals shape: " << invEigVals.rows() << " x " << invEigVals.cols() << std::endl;
    
    // Stack eigvecs at the top
    int p = Ka.rows();
    int n = p + Kab.cols();
    Eigen::MatrixXd phi(n, p);
    phi << eigvecs, (Kab.transpose() * eigvecs * invEigVals);

    return std::make_pair(eigvals, phi);
}

void plotSampledPoints(cv::Mat& I, int nSamples) {
    auto nrows = I.rows;
    auto ncols = I.cols;
    std::vector<int> selected, rest;
    std::tie(selected, rest) = samplePixels(nrows, ncols, nSamples);
    for (auto i : selected) {
        int r, c;
        std::tie(r, c) = to2DCoords(i, ncols);
        cv::circle(I, cv::Point(c, r), 2, cv::Scalar(255, 0, 0), -1);
    }

    std::cout << "# selected: " << selected.size() << std::endl;
}

void sortVector(Eigen::VectorXd& v, std::vector<int>& order) {

}

template <typename T>
cv::Mat filterImage(const cv::Mat& I, std::vector<T>& weights)
{
    cv::Mat Ilab;
    cv::cvtColor(I, Ilab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(Ilab, channels);

    // TODO: Remember to convert back to 8U before merging
    // TODO: Check if value of L depends on original type.
    cv::Mat L = channels[0];
    L.convertTo(L, CV_64F);

    // plotSampledPoints(I.clone(), 10);
    // cv::imshow("sampled", I);
    // cv::waitKey(-1);
    
    std::cout << "Computing kernel weights" << std::endl;
    Eigen::MatrixXd Ka, Kab;
    unsigned int nRowSamples = 7;

    std::vector<int> pixelOrder;
    computeKernelWeights(L, Ka, Kab, pixelOrder, nRowSamples);
    std::cout << "Ka top left corner" << std::endl;
    std::cout << Ka.block<5, 5>(0, 0) << std::endl;

    Eigen::MatrixXd eigvals, phi;
    std::tie(eigvals, phi) = nystromApproximation(Ka, Kab);



    // Visualize eigenvectors. Remember to reshape, sort and convert to CV_8U



    // Eigen::MatrixXd Wa, Wab;
    // std::tie(Wa, Wab) = sinkhornKnopp(phi, eigvals);

    // orthogonalization(Wa, Wab, eigenVectors);

    // TODO: Visualize top eigenvectors

    // Now Wa = Ka, Wab = Kab
    // auto& Wa = Ka;
    // auto& Wab = Kab;
    // auto invSqRootWa = invSqRoot(Wa);
    // auto Q = Wa + invSqRootWa * Wab * Wab.transpose() * invSqRootWa;

    // Eigendecompose Q

    // Compute Vm from eigendecomposition result

    // scale eigenvalues
    // scaleEigenValues(/*eigenvalues ,*/ weights);

    // Compute filter matrix W = Vm * f(Sm) * Vm^T

    // Figure out how to use propagation mask
    // Optionally compute mask
    // Use Mask for propagating edits

    return cv::Mat();
}



