#ifndef FILTER_HPP
#define FILTER_HPP

#include <algorithm>
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
#include <Eigen/Eigenvalues>

#include <Spectra/MatOp/DenseGenMatProd.h>
#include <Spectra/MatOp/DenseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>

namespace nle {
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;
    using DType = double;
    const auto OPENCV_MAT_TYPE = CV_64F;
    const double EPS = 1e-10;
    struct Point {
        int row;
        int col;
    };
    
    inline
    int to1DIndex(int row, int col, int ncols)
    {
        return row * ncols + col;
    }
    
    inline
    std::pair<int, int> to2DCoords(int index, int ncols)
    {
        return std::make_pair(index / ncols, index % ncols);
    }
    
    cv::Mat eigen2opencv(Vec& v, int nrows, int ncols) {
        cv::Mat X(nrows, ncols, OPENCV_MAT_TYPE, v.data());
        return X;
    }
    
    template <typename T>
    Vec opencv2Eigen(const cv::Mat& mat) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> lv;
        lv.resize(mat.total(), 1);
        int k = 0;
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                lv(k) = mat.at<T>(i, j);
                ++k;
            }
        }
        
        return lv;
    }
    
    cv::Mat rescaleForVisualization(const cv::Mat& mat) {
        DType minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);
        cv::Mat rescaledMat = (mat - minVal) / (maxVal - minVal) * 255;
        return rescaledMat;
    }
    
    auto samplePixels(int nrows, int ncols, int nRowSamples, int nColSamples)
    {
        int rowStep = nrows / nRowSamples;
        int colStep = ncols / nColSamples;
        int rowOffset = (rowStep - 1 + (nrows - rowStep * nRowSamples)) / 2;
        int colOffset = (colStep - 1 + (ncols - colStep * nColSamples)) / 2;
        
        std::vector<Point> selected, rest;
        selected.reserve(nRowSamples * nColSamples);
        rest.reserve(nrows * ncols - nRowSamples * nColSamples);
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < ncols; c++) {
                if ((r >= rowOffset && c >= colOffset) && ((r - rowOffset) % rowStep == 0) &&
                    ((c - colOffset) % colStep == 0) && r <= (nrows - rowOffset) &&
                    c <= (ncols - colOffset)) {
                    selected.push_back(Point{r, c});
                } else {
                    rest.push_back(Point{r, c});
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
        
        std::cout << "# row samples: " << nRowSamples << " # col samples: " << nColSamples <<  " # selected: " << selected.size() << " # rest: " << rest.size() << std::endl;
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
    
    void printNegativeEntries(const Mat& mat) {
        for (int j = 0; j < mat.cols(); j++) {
            for (int i = 0; i < mat.rows(); i++) {
                if (mat(i, j) < 0) {
                    std::cout << i << ", " << j << " -ve val: " << mat(i, j) << std::endl;
                }
            }
        }
    }
    
    auto computeKernelWeights(const cv::Mat& mat, int nRowSamples, int nColSamples, DType hx, DType hy)
    {
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
        // Convert Ka and Kab to kernels
        Ka = Ka.array().exp();
        Kab = Kab.array().exp();
        
#ifndef NDEBUG
        assert(Ka.isApprox(Ka.transpose()));
        if (Ka.isApprox(Ka.transpose())) {
            std::cout << "Ka is symmetric" << std::endl;
        }
        else {
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
    
    void reciprocal(Vec& v, DType eps=EPS) {
        for (int i = 0; i < v.rows(); i++) {
            if (std::abs(v(i)) >= eps) {
                v(i) = 1 / v(i);
            }
            else {
                v(i) = 0;
            }
        }
    }
    
    std::pair<Mat, Mat>
    sinkhornKnopp(const Mat& phi, const Mat& eigvals, int maxIter = 20,  DType eps = EPS)
    {
        // TODO: Debug this by testing on small, almost symmetric, matrices.
        
        int n = phi.rows();
        Vec r = Vec::Ones(n, 1);
        Vec c;
        Mat Dphi_t = eigvals.asDiagonal() * phi.transpose();
        for (int i = 0; i < maxIter; i++) {
            c = phi * (Dphi_t * r);
            reciprocal(c, eps);
            assert(c.rows() == phi.rows());
            assert(c.cols() == 1);
            r = phi * (Dphi_t * c);
            reciprocal(r, eps);
        }
        
        // TODO: Check that these quantities are correct
        int p = phi.cols();
        Mat Waab(p, n);
        Mat tmp = (c.replicate(1, p).array() * phi.array()).matrix().transpose();
        for (int i = 0; i < p; i++) {
            Waab.row(i) = r(i) * (eigvals.transpose().array() * phi.row(i).array()).matrix() * tmp;
        }
        
        //    Mat Waab = (r.asDiagonal() * (phi * eigvals.asDiagonal())) * tmp;
        Mat Wa = Waab.leftCols(p);
        Mat Wab = Waab.rightCols(n - p);
        assert(Wa.cols() + Wab.cols() == n);
        
        return std::make_pair(Wa, Wab);
    }
    
    std::pair<Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic>, int>
    invertDiagMatrix(const Mat& mat, DType eps = EPS)
    {
        int numNonZero = 0;
        Mat invMat = mat;
        for (int c = 0; c < mat.cols(); c++) {
            for (int r = 0; r < mat.rows(); r++) {
                if (std::abs(mat(r, c)) < eps) {
                    invMat(r, c) = 0;
                } else {
                    invMat(r, c) = 1.0 / mat(r, c);
                    ++numNonZero;
                }
            }
        }
        
        return std::make_pair(invMat.asDiagonal(), numNonZero);
    }

    auto topkEigenDecomposition(const Mat& M, int nLargest, DType eps=EPS) {
        nLargest = std::min(nLargest, (int) M.rows());

        Spectra::DenseGenMatProd<DType> op_largest(M); 
       // Construct solver object, requesting the top k largest eigenvalues
        Spectra::SymEigsSolver< DType, Spectra::LARGEST_MAGN, Spectra::DenseGenMatProd<DType> >
            solver(&op_largest, nLargest, 2 * nLargest);

        solver.init();
        int nConvergedEigenValues = solver.compute();

        std::cout << "Matrix size: " << M.rows() << " x " << M.cols() << std::endl;
        std::cout << "# converged eigenvalues: " << nConvergedEigenValues << std::endl;
        if(solver.info() != Spectra::SUCCESSFUL) {
            std::cout << "Eigen decomposition NOT successful. Results might be inaccurate." << std::endl;
        }

        Vec eigvals = solver.eigenvalues();
        Mat eigvecs = solver.eigenvectors();

        int r = 0;
        for (r = 0; r < eigvals.size() && eigvals(r) >= eps; ++r);
        if (r < eigvals.size()) {
            eigvecs = eigvecs.leftCols(r);
            eigvals = eigvals.head(r);
        }

        return std::make_pair(eigvecs, eigvals);
    }

    auto eigenDecomposition(const Mat& A, DType eps=EPS) {
        // Compute eigen factorization of a PSD matrix

        // TODO: Replace using solver from Spectra library that computes only top k
        // eigenvectors and eigenvalues.

        Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Vec D = svd.singularValues();
        int rank = svd.rank();
        int r = 0;
        for (r = 0; r < rank && D(r) >= eps; ++r);
        D = svd.singularValues().head(r);
        Mat U = svd.matrixU().leftCols(r);
        Mat V = svd.matrixV().leftCols(r);
        
#ifndef NDEBUG
        std::cout << "Rank: " << svd.rank() << std::endl;
        std::cout << "U shape: " << U.rows() << " x " << U.cols() << std::endl;
        std::cout << "V shape: " << V.rows() << " x " << V.cols() << std::endl;
        std::cout << "D shape: " << D.rows() << " x " << D.cols() << std::endl;
        printNegativeEntries(D);
        Mat mat = V.transpose() * U;
        assert(mat.isIdentity(eps));
        std::cout << "mat: " << std::endl;
        std::cout << mat.topLeftCorner(5, 5) << std::endl;
#endif
        
        return std::make_pair(U, D);
    }
    
    auto nystromApproximation(const Mat& Ka, const Mat& Kab, DType eps = EPS)
    {
        Vec eigvals;
        Mat eigvecs;
        std::tie(eigvecs, eigvals) = eigenDecomposition(Ka);
        
        // Approximate eigenvectors of K from the above eigenvalues and eigenvectors and Kab
        int numNonZero;
        Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic> invEigVals;
        std::tie(invEigVals, numNonZero) = invertDiagMatrix(eigvals, eps);
        
        int p = eigvals.size();
        int n = Ka.cols() + Kab.cols();
        Mat phi(n, p);
        phi << eigvecs, (Kab.transpose() * eigvecs * invEigVals);
        return std::make_pair(eigvals, phi);
    }
    
    auto orthogonalize(const Mat& Wa, const Mat& Wab, int nEigVectors, DType eps=EPS) {
        Mat eigvecs;
        Vec eigvals;
        std::tie(eigvecs, eigvals) = eigenDecomposition(Wa, eps);
        
        Vec invRootEigVals = eigvals;
        reciprocal(invRootEigVals, eps);
        invRootEigVals = invRootEigVals.cwiseSqrt();
        Mat invRootWa = eigvecs * invRootEigVals.asDiagonal() * eigvecs.transpose();
        Mat Q = Wa + invRootWa * Wab * Wab.transpose() * invRootWa;
        
#ifndef NDEBUG
        if (Q.isApprox(Q.transpose(), eps)) {
            std::cout << "Q is symmetric" << std::endl;
        }
        else {
            std::cout << "Q is NOT symmetric" << std::endl;
            std::cout << Q.bottomRightCorner(5, 5) << std::endl;
        }
#endif
        
        Mat Vq;
        Vec Sq;
        std::tie(Vq, Sq) = topkEigenDecomposition(Q, nEigVectors, eps);
        // std::tie(Vq, Sq) = eigenDecomposition(Q, eps);
        Vec invRootSq = Sq;
        reciprocal(invRootSq, eps);
        invRootSq = invRootSq.cwiseSqrt();
        
        // Stack Wa above Wab^T
        Mat tmp(Wa.rows() + Wab.cols(), Wa.cols());
        tmp << Wa, Wab.transpose();
        // Compute approximate eigenvectors of W
        Mat V = tmp * invRootWa * Vq * invRootSq.asDiagonal();
        
        // V contain eigenvectors of W and Sq contains their corresponding eigenvalues
        return std::make_pair(V, Sq);
    }
    
    Mat transformEigenValues(const Mat& eigvals, const std::vector<DType>& weights) {
        int nEigVals = eigvals.rows();
        Mat fS(nEigVals, 1);
        for (int i = 0; i < nEigVals; i++) {
            DType eig = eigvals(i, 0);
            fS(i) = weights[0];
            for (auto k = 1ul; k < weights.size(); k++) {
                fS(i) += (weights[k] - weights[k - 1]) * std::pow(eig, DType(k));
            }
        }
        
        return fS;
    }
    
    template <typename T>
    std::pair<Mat, Eigen::DiagonalMatrix<T, Eigen::Dynamic, Eigen::Dynamic> >
    learnFilter(const cv::Mat& mat, std::vector<T>& weights, int nRowSamples, int nColSamples,
                DType hx, DType hy, int nSinkhornIter, int nEigenVectors=20)
    {
        Mat Ka, Kab;
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P;
        std::tie(P, Ka, Kab) = computeKernelWeights(mat, nRowSamples, nColSamples, hx, hy);
        
        Mat eigvals, phi;
        std::tie(eigvals, phi) = nystromApproximation(Ka, Kab);
        
        Mat Wa, Wab;
        std::tie(Wa, Wab) = sinkhornKnopp(phi, eigvals, nSinkhornIter);
        // Wa = (Wa + Wa.transpose()) / 2;
        
        Mat V, S;
        std::tie(V, S) = orthogonalize(Wa, Wab, nEigenVectors);
        
        V = P * V;
        Mat fS = transformEigenValues(S, weights);
        
#ifndef NDEBUG
        int nEigVals = S.rows();
        int t = std::min(nEigVals, nEigenVectors);
        std::cout << "Top 5 Eigenvalue: " << std::endl;
        std::cout << eigvals.topRows(5) << std::endl;
        std::cout << "S top k of total length: " << S.rows() << std::endl;
        std::cout << S.topRows(t) << std::endl;
        std::cout << "S bottom k" << std::endl;
        std::cout << S.bottomRows(5) << std::endl;
        
        std::cout << "Filtered fS top k" << std::endl;
        std::cout << fS.topRows(10) << std::endl;
        std::cout << "Filtered fS bottom k" << std::endl;
        std::cout << fS.bottomRows(10) << std::endl;
#endif
        
        int nFilters = std::min(nEigenVectors, (int) fS.rows());
        return std::make_pair(V.leftCols(nFilters), fS.topRows(nFilters).asDiagonal());
    }
    
    cv::Mat filterImage(const cv::Mat& mat, std::vector<DType>& weights,
                        int nRowSamples, int nColSamples,
                        DType hx, DType hy,
                        int nSinkhornIter, int nEigenVectors=20)
    {
        cv::Mat II;
        cv::cvtColor(mat, II, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> channels;
        cv::split(II, channels);
        channels[0].convertTo(channels[0], OPENCV_MAT_TYPE);
        
        Mat V;
        Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic> S;
        std::tie(V, S) = learnFilter(channels[0], weights,
                                     nRowSamples, nColSamples, hx, hy,
                                     nSinkhornIter, nEigenVectors);
        
        Vec y = opencv2Eigen<DType>(channels[0]);
        Vec filteredY = V * (S * V.transpose() * y);
        cv::Mat matY = eigen2opencv(filteredY, mat.rows, mat.cols);
        matY.convertTo(matY, CV_8U);
        channels[0] = matY;
        channels[0].convertTo(channels[0], CV_8U);
        
        cv::Mat filteredImage;
        cv::merge(channels, filteredImage);
        cv::cvtColor(filteredImage, filteredImage, cv::COLOR_Lab2BGR);
        return filteredImage;
    }
    
    template <typename T>
    cv::Mat denoiseColorImage(const cv::Mat& mat, std::vector<T>& weights, int nRowSamples, int nColSamples,
                              DType hx, DType hy, int nSinkhornIter)
    {
        cv::Mat II;
        
        // cv::cvtColor(mat, II, cv::COLOR_BGR2YUV);
        cv::cvtColor(mat, II, cv::COLOR_BGR2Lab);
        // cv::cvtColor(mat, Ilab, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(II, channels);
        
        channels[0].convertTo(channels[0], OPENCV_MAT_TYPE);
        channels[1].convertTo(channels[1], OPENCV_MAT_TYPE);
        channels[2].convertTo(channels[2], OPENCV_MAT_TYPE);
        
        Mat V, S;
        std::tie(V, S) = makeFilter(channels[0], weights,
                                    nRowSamples, nColSamples,
                                    hx, hy, nSinkhornIter);
        
        Vec u = opencv2Eigen<DType>(channels[1]);
        Vec v = opencv2Eigen<DType>(channels[2]);
        Vec filteredU = V * (S * V.transpose() * u);
        Vec filteredV = V * (S * V.transpose() * v);
        
        cv::Mat matU = eigen2opencv(filteredU, mat.rows, mat.cols);
        cv::Mat matV = eigen2opencv(filteredV, mat.rows, mat.cols);
        matU.convertTo(matU, CV_8U);
        matV.convertTo(matV, CV_8U);
        channels[0].convertTo(channels[0], CV_8U);
        channels[1] = matU;
        channels[2] = matV;
        
        cv::Mat filteredImage;
        cv::merge(channels, filteredImage);
        
        cv::cvtColor(filteredImage, filteredImage, cv::COLOR_Lab2BGR);
        return filteredImage;
    }
    
}

#endif /* ifndef FILTER_HPP
 */
