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
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>


double kernel(const cv::Mat& I,
              int r, int s,
              int r1, int c1, int r2, int c2,
              double gammaSpatial, double gammaIntensity) {
    
    auto yr = I.at<double>(r, 0);
    auto ys = I.at<double>(s, 0);
    
    // Add in spatial term
    double squareSpatialDist = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2);
    double squareIntensityDist = (yr - ys) * (yr - ys);
    return std::exp(- gammaSpatial * squareSpatialDist - gammaIntensity * squareIntensityDist);
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
    int rowStep = nrows / nRowSamples;
    int colStep = ncols / nColSamples;
    
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
    // Compute single step size for rows and cols by taking into account their
    // ratios
    
    // Reshape channel I into a single column vector.
    // Store the 1D or 2D coordinates of the selected pixels corresponding to Ka
    // Perform stable_partition to move the values of these coordinates to the top
    // of 1D vector.
    // Compute kernel
    // Save these coordinates
    
    auto nrows = I.rows, ncols = I.cols;
    auto nPixels = nrows * ncols;
    
    std::vector<int> selected, rest;
    std::tie(selected, rest) = samplePixels(nrows, ncols, nRowSamples);
    int nSamples = selected.size();

    Ka.resize(nSamples, nSamples);
    Kab.resize(nSamples, nPixels - nSamples);

    std::cout << "nPixels: " << nPixels << std::endl;
    std::cout << "Ka size: " << Ka.rows() << " x " << Ka.cols() << std::endl;
    std::cout << "Kab size: " << Kab.rows() << " x " << Kab.cols() << std::endl;

    double variance = estimateVariance(I);
    double gammaIntensity = 1.0 / variance;
    double gammaSpatial = 0; // 1.0 / 10;
    
    // This is row vector
    cv::Mat II = I.reshape(0, 1);
    std::cout << "II shape: " << II.size() << std::endl;
    // Compute Ka
    int r1, c1, r2, c2;
    for (int i : selected) {
        std::tie(r1, c1) = to2DCoords(i, ncols);
        
        // Compute Ka
        for (int j : selected) {
            std::tie(r2, c2) = to2DCoords(j, ncols);
            Ka(i, j) = kernel(II, i, j, r1, c1, r2, c2,
                              gammaSpatial, gammaIntensity);
        }
        
        // Compute Kab
        for (int j : rest) {
            std::tie(r2, c2) = to2DCoords(j, ncols);
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

template <typename T>
T invSqRoot(T& M)
{

    return M;
}

template <typename T>
void scaleEigenValues(const T& weights)
{
}

// double photoMetricKernel(const cv::Mat)

Eigen::MatrixXd nystromApproximation(const Eigen::MatrixXd& Ka,
                                     const Eigen::MatrixXd& Kab,
                                     double eps=0.00001) {

    // Eigendecomposition of Ka
    Eigen::EigenSolver<Eigen::MatrixXd> es(Ka);

    auto eigvals = es.eigenvalues();
    auto eigvecs = es.eigenvectors();
    
    // TODO: Maybe manually invert eigenvalue matrix and threshold those with eigenvalues < eps to 0
    Eigen::MatrixXd invEigVals = (eigvals.array() + eps).matrix().asDiagonal().inverse();

    // Approximate eigenvectors of K from the above eigenvalues and eigenvectors and Kab
    std::cout << "eigvals shape: " << eigvals.rows() << " x " << eigvals.cols() << std::endl;
    std::cout << "eigvecs shape: " << eigvecs.rows() << " x " << eigvecs.cols() << std::endl;
    std::cout << "invEigVals shape: " << invEigVals.rows() << " x " << invEigVals.cols() << std::endl;
    
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
    cv::Mat L = channels[0];
    L.convertTo(L, CV_64F);
    
    Eigen::MatrixXd Ka, Kab;

    
//    computeKernelWeights(L, Ka, Kab);

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



int main(int argc, char* argv[])
{
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <image> <output> <weight 1> <weight 2> <weight 3> <weight 4>" << std::endl;
        return 0;
    }

    std::string imagePath { argv[1] };
    std::string outputPath { argv[2] };
    std::vector<double> weights;
    for (auto i = 0u; i < 4; ++i) {
        weights.push_back(std::stod(argv[3 + i]));
    }

    std::cout << "Weights: ";
    for (const auto& w : weights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to read file from " << imagePath << std::endl;
        return 0;
    }

    std::cout << image.size() << std::endl;
    
    std::cout << "Size: " << image.size() << std::endl;
    std::cout << "# elements: " << image.size().width * image.size().height << std::endl;
    std::cout << "# elements: " << image.total() << std::endl;
    
//    cv::imshow("original", image);
//    cv::waitKey(0);

    // Always convert image to landscape mode first by rotating it
    // Then rotate it back later. This is to ensure we always sample the shorter side
    // and hence less likely to lose information in the image.

    cv::Mat result = filterImage(image, weights);
    // cv::imwrite(outputPath, result);

    return 0;
}
