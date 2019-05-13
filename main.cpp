#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>


double kernel(const cv::Mat& I, int r, int s, double gamma) {
    auto yr = I.at<double>(r, 0);
    auto ys = I.at<double>(s, 0);
    return std::exp(- gamma * (yr - ys) * (yr - ys));
}

template <typename T>
inline T to1DIndex(T row, T col, T ncols) {
    return row * ncols + col;
}

double estimateVariance(const cv::Mat& I) {
    return 100.0;
}

void computeKernelWeights(const cv::Mat& I,
                          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Ka,
                          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Kab,
                          int nSamples = 50)
{
    auto nrows = I.rows;
    auto ncols = I.cols;
    auto N = nrows * ncols;

    // Crude estimate. We can improve on this later.
    auto step = nrows / nSamples;

    Ka.resize(nSamples, nSamples);
    Kab.resize(nSamples, N - nSamples);

    std::cout << "Finished resizing: N: " << N << std::endl;
    std::cout << "step: " << step << std::endl;
    std::cout << "Ka size: " << Ka.rows() << " x " << Ka.cols() << std::endl;
    std::cout << "Kab size: " << Kab.rows() << " x " << Kab.cols() << std::endl;

    cv::Mat II = I.reshape(0, 1);
    std::cout << "II shape: " << II.size() << std::endl;

    double variance = estimateVariance(I);
    double gamma = 1 / variance;

    // Sample rows and compute kernel matrix
    for (auto k = 0, i = 0; k < nSamples; ++k, i = k * step) {
        auto r = to1DIndex(i, 0, ncols);
        for (auto j = 0; j < nSamples; ++j) {
            auto s = to1DIndex(i, j, ncols);
            std::cout << "r: " << r << " s: " << s <<  " k: " << k << std::endl;
            // auto u = kernel(II, r, s, gamma);
            // std::cout << "u: " << u << std::endl;
            // Ka(k, s) = kernel(II, r, s, gamma);
            Ka(k, s) = 1;
        }

        std::cout << "Finished first block" << std::endl;
        for (auto j = nSamples; j < ncols; ++j) {
            auto s = to1DIndex(i, j, ncols);
            // std::cout << "r: " << r << " s: " << s << std::endl;
            // auto u = kernel(II, r, s, gamma);
            // std::cout << "u: " << u << std::endl;
            // Kab(k, s - nSamples) = kernel(II, r, s, gamma);
            assert(k < nSamples);
            assert(nSamples < N);

            // Kab(k, s - nSamples) = 1;
        }
    }
}

template <typename T>
void sinkhorn(T& Ka, T& Kab)
{
}

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

void nystromApproximation(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Ka,
                          const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Kab) {

    // Eigendecomposition of Ka
    Eigen::EigenSolver<Eigen::MatrixXd> es(Ka);

    auto eigvals = es.eigenvalues();
    auto eigvecs = es.eigenvectors();

    // Approximate eigenvectors of K from the above eigenvalues and eigenvectors and Kab
    std::cout << "eigvals shape: " << eigvals.rows() << " x " << eigvals.cols() << std::endl;


}

template <typename T>
std::vector<T> convertToVec(const cv::Mat& I) {
    std::vector<T> v;
    v.reserve(I.total());
    std::copy(I.begin<T>(), I.end<T>(), std::back_inserter(v));
    
    return v;
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
    
    std::vector<double> v = convertToVec<double>(L);
    assert(v.size() == L.total());
    
    
    
    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ka, Kab;
    
    if (L.isContinuous()) {
        std::cout << "Mat is continuous" << std::endl;
    }
    
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
