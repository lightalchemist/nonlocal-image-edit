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

    return 1.0;

    // auto yr = I.at<double>(r, 0);
    // auto ys = I.at<double>(s, 0);
    // auto yr = I.at<double>(0, r);
    // auto ys = I.at<double>(0, s);
    
    // Add in spatial term
    // double squareSpatialDist = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2);
    // double squareIntensityDist = (yr - ys) * (yr - ys);
    // return std::exp(- spatialScale * squareSpatialDist - intensityScale * squareIntensityDist);
}

template <typename T>
inline T to1DIndex(T row, T col, T ncols) {
    return row * ncols + col;
}

template <typename T>
inline auto to2DCoords(T index, T ncols) {
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

auto samplePixels(int nrows, int ncols, int nRowSamples) {
    float ratio = static_cast<float>(ncols) / static_cast<float>(nrows);
    int nColSamples =  std::floor(std::fmax(ratio * nRowSamples, 1.0));
    
    // Crude estimate. We can improve on this later by having initial row and col offset.
    int rowStep = nrows / (nRowSamples + 1);
    int colStep = ncols / (nColSamples + 1);
    
    int nPixels = nrows * ncols;
    int nSamples = nRowSamples * nColSamples;
    
    std::vector<int> selected;
    std::vector<int> rest;
    selected.reserve(nSamples);
    rest.reserve(nPixels - nSamples);
    
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            if ((r % rowStep) == 0 && (c % colStep == 0)) {
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
                          int nRowSamples = 10)
{

    // Reshape channel I into a single column vector.
    // Store the 1D or 2D coordinates of the selected pixels corresponding to Ka
    // Perform stable_partition to move the values of these coordinates to the top
    // of 1D vector.
    // Compute kernel
    // Save these coordinates
    
    auto nrows = I.rows, ncols = I.cols;
    auto nPixels = I.total();
    assert(nPixels == nrows * ncols);
    
    std::vector<int> selected, rest;
    std::tie(selected, rest) = samplePixels(nrows, ncols, nRowSamples);
    int nSamples = selected.size();

    Ka.resize(nSamples, nSamples);
    Kab.resize(nSamples, nPixels - nSamples);

    std::cout << "nPixels: " << nPixels << std::endl;
    std::cout << "Ka size: " << Ka.rows() << " x " << Ka.cols() << std::endl;
    std::cout << "Kab size: " << Kab.rows() << " x " << Kab.cols() << std::endl;

    // TODO: Tune this
    // double variance = estimateVariance(I);
    double variance = 100;
    double gammaIntensity = 1.0 / variance;
    double gammaSpatial = 0; // 1.0 / 10;
    
    // This is row vector
    cv::Mat II = I.reshape(0, 1);
    std::cout << "II shape: " << II.size() << std::endl;
    // Compute Ka
    int r1, c1, r2, c2;
    int k = 0;
    for (int i = 0; i < selected.size(); ++i) {
        for (int j = 0; j < selected.size(); ++j) {

        }
    }

    
    for (int i : selected) {
        std::tie(r1, c1) = to2DCoords(i, ncols);
        for (int j : selected) {
            std::tie(r2, c2) = to2DCoords(j, ncols);
            assert(0 <= i && i < Ka.rows());
            assert(0 <= j && j < Ka.cols());
            Ka(i, j) = kernel(II, i, j, r1, c1, r2, c2,
                              gammaSpatial, gammaIntensity);
        }

        ++k;
    }


        // Compute Kab
    for (int i : selected) {
        for (int j : rest) {
            std::tie(r2, c2) = to2DCoords(j, ncols);
            assert(i < Kab.rows());
            assert(j < Kab.cols());
            Kab(i, j) = kernel(II, i, j, r1, c1, r2, c2,
                               gammaSpatial, gammaIntensity);
        }
    }
}

//void sinkhorn(Eigen::MatrixXd& phi, Eigen::MatrixXd& eigvals,
//              int maxIter=20)
//{
//    int n = phi.rows();
//    auto r = Eigen::ArrayXXf::Ones(n, 1);
//
//    for (int i = 0; i < maxIter; i++) {
//
//    }
//
//}

// template <typename T>
// T invSqRoot(T& M)
// {

//     return M;
// }

template <typename T>
void scaleEigenValues(const T& weights)
{
}

template <typename T>
auto robustInvertDiagMatrix(const T& A, double eps=0.00001) {
    Eigen::MatrixXd invMat = A;
    for (int c = 0; c < A.cols(); c++) {
        for (int r = 0; r < A.rows(); r++) {
            if (A(r, c) < eps) {
                invMat(r, c) = 0;
            }
            else {
                invMat(r, c) = 1.0 / A(r, c);
            }
        }
    }

    return A.asDiagonal();
}

Eigen::MatrixXd nystromApproximation(const Eigen::MatrixXd& Ka,
                                     const Eigen::MatrixXd& Kab,
                                     double eps=0.00001) {

    // Eigendecomposition of Ka
    Eigen::EigenSolver<Eigen::MatrixXd> es(Ka);

    // TODO: Check this. This results in a conversion to MatrixXd?
    Eigen::MatrixXd eigvals = es.eigenvalues().real();
    Eigen::MatrixXd eigvecs = es.eigenvectors().real();
    auto invEigVals = robustInvertDiagMatrix(eigvals, eps);

    // Approximate eigenvectors of K from the above eigenvalues and eigenvectors and Kab
    std::cout << "eigvals shape: " << eigvals.rows() << " x " << eigvals.cols() << std::endl;
    std::cout << "eigvecs shape: " << eigvecs.rows() << " x " << eigvecs.cols() << std::endl;
    std::cout << "invEigVals shape: " << invEigVals.rows() << " x " << invEigVals.cols() << std::endl;
    
    // Stack eigvecs at the top
    int p = Ka.rows();
    int n = eigvecs.rows();
    Eigen::MatrixXd phi(n, p);
    phi << eigvecs, (Kab.transpose() * eigvecs * invEigVals);

    return phi;
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
    
    std::cout << "Computing kernel weights" << std::endl;
    Eigen::MatrixXd Ka, Kab;
    computeKernelWeights(L, Ka, Kab);
    // nystromApproximation(Ka, Kab);
    // sinkhorn(Ka, Kab);

    // orthogonalization(Wa, Wab, eigenVectors);

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



