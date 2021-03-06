#include "filter.hpp"
#include "utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#ifdef USE_SPECTRA
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#endif

using nle::DType;
using nle::EPS;
using nle::Mat;
using nle::NLEFilter;
using nle::OPENCV_MAT_TYPE;
using nle::Point;
using nle::Vec;


namespace nle 
{
    cv::Mat rescaleForVisualization(const cv::Mat& mat)
    {
        DType minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);
        cv::Mat rescaledMat = (mat - minVal) / (maxVal - minVal) * 255;
        return rescaledMat;
    }

    int inplaceReciprocal(Vec& v, DType eps = EPS)
    {
        int nnz = 0;
        for (int i = 0; i < v.rows(); i++) {
            if (std::abs(v(i)) >= eps) {
                v(i) = 1 / v(i);
                ++nnz;
            } else {
                v(i) = 0;
            }
        }
        return nnz;
    }

    auto samplePixels(int nrows, int ncols, int nRowSamples, int nColSamples)
    {
        const int rowStep = nrows / nRowSamples;
        const int colStep = ncols / nColSamples;
        const int rowOffset = (rowStep - 1 + (nrows - rowStep * nRowSamples)) / 2;
        const int colOffset = (colStep - 1 + (ncols - colStep * nColSamples)) / 2;

        std::vector<Point> selected, rest;
        selected.reserve(nRowSamples * nColSamples);
        rest.reserve(nrows * ncols - nRowSamples * nColSamples);
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < ncols; c++) {
                if ((r >= rowOffset && c >= colOffset) && r <= (nrows - rowOffset) && c <= (ncols - colOffset) &&
                    ((r - rowOffset) % rowStep == 0) && 
                    ((c - colOffset) % colStep == 0) 
                    ) {
                    selected.push_back(Point{ r, c });
                } else {
                    rest.push_back(Point{ r, c });
                }
            }
        }

        return std::make_pair(selected, rest);
    }

    void plotSampledPoints(cv::Mat& mat, int nRowSamples, int nColSamples)
    {
        int nrows = mat.rows;
        int ncols = mat.cols;
        std::vector<Point> selected, rest;
        std::tie(selected, rest) = samplePixels(nrows, ncols, nRowSamples, nColSamples);
        for (Point p : selected) {
            int r = p.row, c = p.col;
            cv::circle(mat, cv::Point(c, r), 2, cv::Scalar(255, 0, 0), -1);
        }
    }

    DType kernel(const cv::Mat& mat, const Point& p1, const Point& p2,
                 DType spatialWeight, DType photometricWeight)
    {
        DType yr = mat.at<DType>(p1.row, p1.col);
        DType ys = mat.at<DType>(p2.row, p2.col);
        DType squareSpatialDist = (p1.row - p2.row) * (p1.row - p2.row) + (p1.col - p2.col) * (p1.col - p2.col);
        DType squareIntensityDist = (yr - ys) * (yr - ys);
        return std::exp(-spatialWeight * squareSpatialDist - photometricWeight * squareIntensityDist);
    }

    DType negativeWeightedDistance(const cv::Mat& mat, const Point& p1, const Point& p2,
                                   DType spatialWeight, DType photometricWeight)
    {
        DType yr = mat.at<DType>(p1.row, p1.col);
        DType ys = mat.at<DType>(p2.row, p2.col);
        DType squareSpatialDist = (p1.row - p2.row) * (p1.row - p2.row) + (p1.col - p2.col) * (p1.col - p2.col);
        DType squareIntensityDist = (yr - ys) * (yr - ys);
        return -spatialWeight * squareSpatialDist - photometricWeight * squareIntensityDist;
    }

    std::tuple<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>, Mat, Mat> 
    computeKernel(const cv::Mat& mat, int nRowSamples, int nColSamples, DType hx, DType hy)
    {
        if (nRowSamples > mat.rows || nColSamples > mat.cols) {
            throw std::runtime_error("Number of samples per row and col must be <= that of image.");
        }

        int nPixels = mat.total();
        std::vector<Point> selected, rest;
        std::tie(selected, rest) = samplePixels(mat.rows, mat.cols, nRowSamples, nColSamples);
        int nSamples = selected.size();
        Mat Ka(nSamples, nSamples);
        Mat Kab(nSamples, nPixels - nSamples);

        DType photometricWeight = 1.0 / (hy * hy);
        DType spatialWeight = 1.0 / (hx * hx);
        for (auto i = 0u; i < selected.size(); ++i) {
            // Ka
            Point p1 = selected[i];
            for (auto j = i; j < selected.size(); ++j) {
                auto val = negativeWeightedDistance(mat, p1, selected[j], spatialWeight, photometricWeight);
                Ka(i, j) = val;
                Ka(j, i) = val;
            }
            // Kab
            for (auto j = 0u; j < rest.size(); j++) {
                Kab(i, j) = negativeWeightedDistance(mat, p1, rest[j], spatialWeight, photometricWeight);
            }
        }

        Ka.array() = Ka.array().exp();
        Kab.array() = Kab.array().exp();

#ifndef NDEBUG
        assert(Ka.isApprox(Ka.transpose()));
        if (Ka.isApprox(Ka.transpose())) {
            std::cout << "Ka is symmetric" << std::endl;
        } else {
            std::cout << "Ka is NOT symmetric" << std::endl;
        }
#endif

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(nPixels);
        for (auto i = 0u; i < selected.size(); ++i) {
            Point p = selected[i];
            P.indices()[i] = to1DIndex(p.row, p.col, mat.cols);
        }
        for (auto j = 0u; j < rest.size(); ++j) {
            Point p = rest[j];
            P.indices()[j + selected.size()] = to1DIndex(p.row, p.col, mat.cols);
        }

        return std::make_tuple(P, Ka, Kab);
    }

#ifdef USE_SPECTRA
    auto topkEigenDecomposition(const Mat& M, int nLargest, DType eps = EPS)
    {
        nLargest = std::min(nLargest, static_cast<int>(M.rows() - 1));
        assert(nLargest > 0);
        int ncv = std::min(2 * nLargest, static_cast<int>(M.rows()));
        Spectra::DenseGenMatProd<DType> op_largest(M);
        Spectra::SymEigsSolver<DType, Spectra::LARGEST_MAGN, Spectra::DenseGenMatProd<DType>>
            solver(&op_largest, nLargest, ncv);
        solver.init();
        int nConvergedEigenValues = solver.compute();
        if (solver.info() != Spectra::SUCCESSFUL) {
            std::cerr << "# converged eigenvalues: " << nConvergedEigenValues << std::endl;
            std::cerr << "Eigen decomposition NOT successful. Results might be inaccurate." << std::endl;
        }

        Vec eigvals = solver.eigenvalues();
        Mat eigvecs = solver.eigenvectors();

        // Keep only eigenvalues larger than threshold
        int r = 0;
        for (r = 0; r < eigvals.size() && eigvals(r) >= eps; ++r)
            ;
        if (r < eigvals.size()) {
            Mat evecs = eigvecs.leftCols(r);
            Vec evals = eigvals.head(r);
            return std::make_pair(evecs, evals);
        } else {
            return std::make_pair(eigvecs, eigvals);
        }
    }
#endif

    // TODO: Implement move semantic version of this to improve
    // memory performance of orthogonalize
    std::pair<Mat, Vec> 
    eigenDecomposition(const Mat& M, DType eps)
    {
        Eigen::SelfAdjointEigenSolver<nle::Mat> es;
        es.compute(M);
        Vec D = es.eigenvalues().reverse();
        Mat U = es.eigenvectors().rowwise().reverse();

        // Keep only valid number of eigenvectors and their eigenvalues
        int r = 0;
        for (r = 0; r < D.size() && D(r) >= eps; ++r);
        D = D.head(r).eval();
        U = U.leftCols(r).eval();

#ifndef NDEBUG
        // TODO: Replace this with proper logger
        std::cout << "U shape: " << U.rows() << " x " << U.cols() << std::endl;
        std::cout << "D shape: " << D.rows() << " x " << D.cols() << std::endl;
        std::cout << "Smallest k eigenvalues" << std::endl;
        int k = std::min(5, r);
        std::cout << D.tail(k) << std::endl;
#endif

        return std::make_pair(U, D);
    }

    std::pair<Mat, Mat>
    sinkhorn(const Mat& phi, const Vec& eigvals, int maxIter)
    {
        int n = phi.rows();
        Vec r = Vec::Ones(n, 1);
        Vec c;

        Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic> D = eigvals.asDiagonal();
        for (int i = 0; i < maxIter; i++) {
            c = phi * (D * (phi.transpose() * r));
            inplaceReciprocal(c);
            assert(c.rows() == phi.rows());
            assert(c.cols() == 1);
            r = phi * (D * (phi.transpose() * c));
            inplaceReciprocal(r);
        }

        int p = phi.cols();
        Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic> R = r.head(p).asDiagonal();
        Mat Wa = (R * (phi.topRows(p) * D)) * (c.head(p).replicate(1, p).array() * phi.topRows(p).array()).matrix().transpose();
        Mat Wab = (R * (phi.topRows(p) * D)) * (c.tail(n - p).replicate(1, p).array() * phi.bottomRows(n - p).array()).matrix().transpose();

        assert(Wa.cols() + Wab.cols() == phi.rows());
        return std::make_pair(Wa, Wab);
    }

    // TODO: Implement move semantic version of this to improve memory performance
    std::pair<Vec, Mat> 
    nystromApproximation(const Mat& Ka, const Mat& Kab)
    {
        Vec eigvals;
        Mat eigvecs;
        std::tie(eigvecs, eigvals) = nle::eigenDecomposition(Ka);

        // Approximate eigenvectors of K from the above eigenvalues and eigenvectors and Kab
        Vec tmp = eigvals;
        int numNonZero = inplaceReciprocal(tmp);

        Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic> invEigVals = tmp.head(numNonZero).asDiagonal();
        eigvecs = eigvecs.leftCols(numNonZero).eval();

        eigvals = eigvals.head(numNonZero).eval();

        int n = Ka.cols() + Kab.cols();
        Mat phi(n, eigvecs.cols());
        phi << eigvecs, (Kab.transpose() * eigvecs * invEigVals);

        assert(eigvals.size() == phi.cols());

        return std::make_pair(eigvals, phi);
    }

    std::pair<Mat, Vec> 
    orthogonalize(const Mat& Wa, const Mat& Wab, int nEigVectors, DType eps)
    {
        Mat eigvecs;
        Vec eigvals;
        std::tie(eigvecs, eigvals) = nle::eigenDecomposition(Wa);

        Vec invRootEigVals = eigvals;
        inplaceReciprocal(invRootEigVals);
        invRootEigVals = invRootEigVals.cwiseSqrt();
        Mat invRootWa = eigvecs * invRootEigVals.asDiagonal() * eigvecs.transpose();
        // NOTE: the parentheses around Wab and Wab^T is crucial for ensuring that expression
        // gets evaluated first to become a small p x p matrix, otherwise the compiler might
        // generate two copies intermediate matrices of size p x n and n x p respectively.
        Mat Q = Wa + invRootWa * (Wab * Wab.transpose()) * invRootWa;

#ifndef NDEBUG
        if (Q.isApprox(Q.transpose(), eps)) {
            std::cout << "Q is symmetric" << std::endl;
        } else {
            std::cout << "Q is NOT symmetric" << std::endl;
            std::cout << Q.bottomRightCorner(5, 5) << std::endl;
        }
#endif

        Mat Vq;
        Vec Sq;

#ifdef USE_SPECTRA
        std::tie(Vq, Sq) = topkEigenDecomposition(Q, nEigVectors);
#else
        std::tie(Vq, Sq) = nle::eigenDecomposition(Q);
        int k = std::min(nEigVectors, static_cast<int>(Vq.cols()));
        Vq = Vq.leftCols(k).eval();
        Sq = Sq.head(k).eval();
#endif

        Vec invRootSq = Sq;
        inplaceReciprocal(invRootSq);
        invRootSq = invRootSq.cwiseSqrt();

        // Stack Wa above Wab^T
        Mat tmp(Wa.rows() + Wab.cols(), Wa.cols());
        tmp << Wa, Wab.transpose();
        // Compute approximate eigenvectors of W
        Mat V = tmp * invRootWa * Vq * invRootSq.asDiagonal();

        // V contain eigenvectors of W and Sq contains their corresponding eigenvalues
        return std::make_pair(V, Sq);
    }
}

Vec transformEigenValues(const Vec& eigvals, const std::vector<DType>& weights)
{
    int nEigVals = eigvals.size();
    Vec fS(nEigVals);
    for (int i = 0; i < nEigVals; i++) {
        DType eig = eigvals(i);
        fS(i) = weights[0];
        for (auto k = 1ul; k < weights.size(); k++) {
            fS(i) += (weights[k] - weights[k - 1]) * std::pow(eig, DType(k));
        }
    }

    return fS;
}

cv::Mat NLEFilter::denoise(const cv::Mat& image, DType k, int sigmaColor, int sigmaSpace) const
{
    if (image.channels() != 3) {
        throw std::runtime_error("Can only enchance RGB image.");
    }

    if (image.total() != m_eigvecs.rows()) {
        throw std::runtime_error("Cannot apply filter on image with different size from the image filter was trained on.");
    }

    cv::Mat II;
    // cv::cvtColor(image, II, cv::COLOR_BGR2YUV);
    cv::cvtColor(image, II, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(II, channels);

    cv::Mat bfImage;
    cv::bilateralFilter(image, bfImage, -1, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);

    cv::Mat originalY = channels[0];

    cv::Mat Y;
    cv::bilateralFilter(channels[0], Y, -1, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
    channels[0] = Y;

    channels[0].convertTo(channels[0], OPENCV_MAT_TYPE);
    channels[1].convertTo(channels[1], OPENCV_MAT_TYPE);
    channels[2].convertTo(channels[2], OPENCV_MAT_TYPE);

    Vec teigvals = m_eigvals;
    for (int i = 0; i < teigvals.size(); i++) {
        DType eval = teigvals(i);
        eval = std::min(eval, 1.0);
        std::cout << "eig " << i << " val: " << eval << std::endl;
        teigvals(i) = std::pow(eval, k);
        // teigvals(i) = 1 - std::pow(1 - eval, k + 1);
    }

    // channels[0] = apply(channels[0], teigvals);
    channels[1] = apply(channels[1], teigvals);
    channels[2] = apply(channels[2], teigvals);

    channels[0] = cv::max(channels[0], 0);
    channels[0] = cv::min(channels[0], 255);
    channels[0].convertTo(channels[0], CV_8U);
    channels[1] = cv::max(channels[1], 0);
    channels[1] = cv::min(channels[1], 255);
    channels[1].convertTo(channels[1], CV_8U);
    channels[2] = cv::max(channels[2], 0);
    channels[2] = cv::min(channels[2], 255);
    channels[2].convertTo(channels[2], CV_8U);

    cv::imshow("luminosity channel", channels[0]);
    cv::imshow("Original luminosity channel", originalY);
    cv::imshow("Bilateral filtered", bfImage);

    cv::Mat filteredImage;
    cv::merge(channels, filteredImage);
    // cv::cvtColor(filteredImage, filteredImage, cv::COLOR_YUV2BGR);
    cv::cvtColor(filteredImage, filteredImage, cv::COLOR_Lab2BGR);
    return filteredImage;
}

cv::Mat NLEFilter::enhance(const cv::Mat& image, const std::vector<DType>& weights) const
{
    if (image.channels() != 3) {
        throw std::runtime_error("Can only enhance RGB image.");
    }

    if (image.total() != m_eigvecs.rows()) {
        throw std::runtime_error("Cannot apply filter on image with different size from the image filter was trained on.");
    }

    cv::Mat II;
    cv::cvtColor(image, II, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(II, channels);
    channels[0].convertTo(channels[0], OPENCV_MAT_TYPE);

    Vec fS = transformEigenValues(m_eigvals, weights);
    assert(fS.size() == m_eigvals.size());
    assert(fS.size() == m_eigvecs.cols());
    channels[0] = apply(channels[0], fS);

    // TODO: Check if we can do this inplace
    channels[0] = cv::max(channels[0], 0);
    channels[0] = cv::min(channels[0], 255);
    channels[0].convertTo(channels[0], CV_8U);

    cv::Mat filteredImage;
    cv::merge(channels, filteredImage);
    cv::cvtColor(filteredImage, filteredImage, cv::COLOR_Lab2BGR);

    return filteredImage;
}

cv::Mat NLEFilter::apply(const cv::Mat& channel, const Vec& transformedEigVals) const
{
    if (channel.total() != m_eigvecs.rows()) {
        throw std::runtime_error("Number of values in channel must match that of training image.");
    }

    const DType* p = channel.ptr<DType>();
    Eigen::Map<const Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    c(p, channel.total(), 1); 

    // Vec c = opencv2eigen<DType>(channel);
    Vec filtered = m_eigvecs * (transformedEigVals.asDiagonal() * m_eigvecs.transpose() * c);
    return eigen2opencv(filtered, channel.rows, channel.cols);
}

cv::Mat getLuminanceChannel(const cv::Mat& image)
{
    cv::Mat Ilab;
    cv::cvtColor(image, Ilab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(Ilab, channels);
    channels[0].convertTo(channels[0], OPENCV_MAT_TYPE);
    cv::Mat luminosity = channels[0];
    return luminosity;
}

cv::Mat getYChannel(const cv::Mat& image)
{
    cv::Mat yuv;
    cv::cvtColor(image, yuv, cv::COLOR_BGR2YUV);
    std::vector<cv::Mat> channels;
    cv::split(yuv, channels);
    return channels[0];
}

void NLEFilter::trainFilter(const cv::Mat& channel, int nRowSamples, int nColSamples,
                            DType hx, DType hy, int nSinkhornIter, int nEigenVectors)
{
    std::cout << "Computing kernel" << std::endl;
    Mat Ka, Kab;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P;
    std::tie(P, Ka, Kab) = computeKernel(channel, nRowSamples, nColSamples, hx, hy);

    std::cout << "Nystrom approximation" << std::endl;
    Vec eigvals;
    Mat phi;
    std::tie(eigvals, phi) = nystromApproximation(Ka, Kab);

    std::cout << "Sinkhorn" << std::endl;
    Mat Wa, Wab;
    std::tie(Wa, Wab) = sinkhorn(phi, eigvals, nSinkhornIter);
    // Wa = (Wa + Wa.transpose()).eval() / 2;

    std::cout << "Orthogonalize" << std::endl;
    std::tie(m_eigvecs, m_eigvals) = orthogonalize(Wa, Wab, nEigenVectors);

    // Permute values back into correct position
    m_eigvecs = (P * m_eigvecs).eval();

    for (int i = 0; i < std::min(nEigenVectors, 5); i++) {
        Vec v = m_eigvecs.col(i);
        std::cout << "Eigvec " << i << " eigval: " << m_eigvals(i) << " minCoeff: " << v.minCoeff() << " maxCoeff: " << v.maxCoeff() << std::endl;
        cv::Mat m = eigen2opencv(v, channel.rows, channel.cols);
        m = rescaleForVisualization(m);
        m.convertTo(m, CV_8U);
        cv::imshow("image" + std::to_string(i), m);
    }
}

void NLEFilter::trainForEnhancement(const cv::Mat& image, int nRowSamples, int nColSamples,
                                    DType hx, DType hy, int nSinkhornIter, int nEigenVectors)
{
    cv::Mat luminance = getLuminanceChannel(image);
    trainFilter(luminance, nRowSamples, nColSamples, hx, hy, nSinkhornIter, nEigenVectors);
}

void NLEFilter::trainForDenoise(const cv::Mat& image, int nRowSamples, int nColSamples,
                                DType hx, DType hy, int nSinkhornIter, int nEigenVectors,
                                int sigmaColor, int sigmaSpace)
{
    // cv::Mat Y = getYChannel(image);
    // cv::Mat Y = getLuminanceChannel(image);
    cv::Mat Ilab;
    cv::cvtColor(image, Ilab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(Ilab, channels);
    cv::Mat L = channels[0];
    channels.clear();

    cv::Mat denoised;
    cv::bilateralFilter(L, denoised, -1, sigmaColor, sigmaSpace, cv::BORDER_DEFAULT);
    denoised.convertTo(denoised, OPENCV_MAT_TYPE);
    trainFilter(denoised, nRowSamples, nColSamples, hx, hy, nSinkhornIter, nEigenVectors);
}
