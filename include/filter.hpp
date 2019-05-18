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
// #include <opencv2/core/eigen.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/IterativeLinearSolvers>
// #include <Eigen/Sparse>
#include <Eigen/Eigenvalues>


const double EPS = 1e-19;

double kernel(const cv::Mat& I,
              int r1, int c1, int r2, int c2,
              double spatialScale, double intensityScale)
{
    double yr = I.at<double>(r1, c1);
    double ys = I.at<double>(r2, c2);

    // Add in spatial term
    // double squareSpatialDist = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2);
    double squareIntensityDist = (yr - ys) * (yr - ys);
    // return std::exp(-spatialScale * squareSpatialDist - intensityScale * squareIntensityDist);
    return std::exp(- intensityScale * squareIntensityDist);
}

inline int to1DIndex(int row, int col, int ncols)
{
    return row * ncols + col;
}

inline std::pair<int, int> to2DCoords(int index, int ncols)
{
    return std::make_pair(index / ncols, index % ncols);
}

double estimateVariance(const cv::Mat& I)
{
    return 100.0;
}

// template <typename T>
// std::vector<T> convertToVec(const cv::Mat& I)
// {
//     std::vector<T> v;
//     v.reserve(I.total());
//     std::copy(I.begin<T>(), I.end<T>(), std::back_inserter(v));
//
//     return v;
// }

auto samplePixels(int nrows, int ncols, int nRowSamples)
{
    float ratio = static_cast<float>(ncols) / static_cast<float>(nrows);
    int nColSamples = std::floor(std::fmax(ratio * nRowSamples, 1.0));

    int nPixels = nrows * ncols;
    int nSamples = nRowSamples * nColSamples;

    int rowStep = nrows / (nRowSamples);
    int colStep = ncols / (nColSamples);
    int rOffset = (rowStep + (nrows - nRowSamples * rowStep)) / 2;
    int cOffset = (colStep + (ncols - nColSamples * colStep)) / 2;

    std::cout << "# row samples: " << nRowSamples << " # col samples: " << nColSamples << std::endl;

    std::vector<int> selected;
    std::vector<int> rest;
    selected.reserve(nSamples);
    rest.reserve(nPixels - nSamples);

    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            if ((r >= rOffset && c >= cOffset) && ((r - rOffset) % rowStep == 0) && ((c - cOffset) % colStep == 0)) {
                selected.push_back(to1DIndex(r, c, ncols));
            } else {
                rest.push_back(to1DIndex(r, c, ncols));
            }
        }
    }

    return std::make_pair(selected, rest);
}

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
computeKernelWeights(const cv::Mat& I,
                     Eigen::MatrixXd& Ka,
                     Eigen::MatrixXd& Kab,
                     // std::vector<int>& pixelOrder,
                     int nRowSamples = 10)
{
    int nrows = I.rows, ncols = I.cols;
    int nPixels = I.total();
    assert(nPixels == nrows * ncols);

    std::vector<int> selected, rest;
    std::tie(selected, rest) = samplePixels(nrows, ncols, nRowSamples);
    int nSamples = selected.size();


    Ka.resize(nSamples, nSamples);
    Kab.resize(nSamples, nPixels - nSamples);

    std::cout << "Ka size: " << Ka.rows() << " x " << Ka.cols() << std::endl;
    std::cout << "Kab size: " << Kab.rows() << " x " << Kab.cols() << std::endl;

    // TODO: Tune this
    // double variance = estimateVariance(I);
    double variance = 1000;
    double gammaIntensity = 1.0 / variance;
    double gammaSpatial = 0; // 1.0 / 10;

    // This is row vector
    // cv::Mat II = I.reshape(0, 1);
    // std::cout << "II shape: " << II.size() << std::endl;

    // Compute Ka
    int r1, c1, r2, c2;
    for (auto i = 0u; i < selected.size(); ++i) {
        std::tie(r1, c1) = to2DCoords(selected[i], ncols);
        for (auto j = i; j < selected.size(); ++j) {
            std::tie(r2, c2) = to2DCoords(selected[j], ncols);
            // auto val = kernel(II, selected[i], selected[j], r1, c1, r2, c2, gammaSpatial, gammaIntensity);
            auto val = kernel(I, r1, c1, r2, c2, gammaSpatial, gammaIntensity);
            Ka(i, j) = val;
            Ka(j, i) = val;
        }

        for (auto j = 0u; j < rest.size(); j++) {
            std::tie(r2, c2) = to2DCoords(rest[j], ncols);
            // Kab(i, j) = kernel(II, selected[i], rest[j], r1, c1, r2, c2, gammaSpatial, gammaIntensity);
            Kab(i, j) = kernel(I, r1, c1, r2, c2, gammaSpatial, gammaIntensity);
        }
    }

    // for (auto i = 0u; i < selected.size(); ++i) {
    //     std::tie(r1, c1) = to2DCoords(i, ncols);
    //     for (auto j = 0u; j < selected.size(); ++j) {
    //         std::tie(r2, c2) = to2DCoords(j, ncols);
    //         Ka(i, j) = kernel(II, selected[i], selected[j],
    //                           r1, c1, r2, c2,
    //                           gammaSpatial, gammaIntensity);
    //     }
    // }

    for (auto i = 0u; i < Ka.rows(); i++) {
        for (auto j = 0u; j < Ka.cols(); j++) {
            if (Ka(i, j) < 0) {
                std::cout << "Ka(" << i << ", " << j << "): " << Ka(i, j) << std::endl;
                throw std::runtime_error("Ka contains negative entries");
            }
        }
    }

    // Check that Ka is symmetric
    std::cout << "Ka rows: " << Ka.rows() << " x " << Ka.cols() << std::endl;
    assert(Ka.isApprox(Ka.transpose()));
    if (Ka.isApprox(Ka.transpose())) {
        std::cout << "Ka is symmetric" << std::endl;
    }
    else {
        std::cout << "Ka is NOT symmetric" << std::endl;
    }

    // for (auto i = 0u; i < selected.size(); i++) {
    //     std::tie(r1, c1) = to2DCoords(i, ncols);
    //     for (auto j = 0u; j < rest.size(); j++) {
    //         std::tie(r2, c2) = to2DCoords(j, ncols);
    //         Kab(i, j) = kernel(II, selected[i], rest[j],
    //                            r1, c1, r2, c2,
    //                            gammaSpatial, gammaIntensity);
    //     }
    // }


    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(nPixels);
    int i = 0;
    for (; i < selected.size(); ++i) {
        P.indices()[i] = selected[i];
    }
    for (int j = 0; j < rest.size(); ++i, ++j) {
        P.indices()[i] = rest[j];
    }

    return P;

    // pixelOrder.reserve(nPixels);
    // pixelOrder.insert(pixelOrder.end(), selected.begin(), selected.end());
    // pixelOrder.insert(pixelOrder.end(), rest.begin(), rest.end());


}

void reciprocal(Eigen::MatrixXd& mat, double eps=EPS) {
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            if (std::abs(mat(i, j)) >= eps) {
                mat(i, j) = 1 / mat(i, j);
            }
            else {
                mat(i, j) = 0;
            }
        }
    }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> 
sinkhornKnopp(const Eigen::MatrixXd& phi,
              const Eigen::MatrixXd& eigvals,
              int maxIter = 20,  double eps = EPS)
{
    // TODO: Debug this by testing on small, almost symmetric, matrices.

    int n = phi.rows();
    int k = phi.cols();
    Eigen::MatrixXd r = Eigen::VectorXd::Ones(n, 1);
    Eigen::MatrixXd c; 

    Eigen::MatrixXd Dphi_t = eigvals.asDiagonal() * phi.transpose();
    for (int i = 0; i < maxIter; i++) {
        // c = phi * (eigvals.array() * (phi.transpose() * r).array()).matrix();
        c = phi * (Dphi_t * r);
        reciprocal(c, eps);

        assert(c.rows() == phi.rows());

        // r = phi * (eigvals.array() * (phi.transpose() * c).array()).matrix();
        r = phi * (Dphi_t * c);
        reciprocal(r, eps);
    }

    // TODO: Check that these quantities are correct
    int p = phi.cols();
    Eigen::MatrixXd Waab(p, n);
    Eigen::MatrixXd tmp = (c.replicate(1, p).array() * phi.array()).matrix().transpose();
    for (int i = 0; i < p; i++) {
        Waab.row(i) = r(i) * (eigvals.transpose().array() * phi.row(i).array()).matrix() * tmp;
    }

    Eigen::MatrixXd Wa = Waab.leftCols(p);
    Eigen::MatrixXd Wab = Waab.rightCols(n - p);
    assert(Wa.cols() + Wab.cols() == n);

    return std::make_pair(Wa, Wab);
}

std::pair<Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic>, int> 
invertDiagMatrix(const Eigen::MatrixXd& mat, double eps = EPS)
{
    int numNonZero = 0;
    Eigen::MatrixXd invMat = mat;
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

    // return std::make_pair(invMat.topLeftCorner(numNonZero, numNonZero).asDiagonal(),
    //                       numNonZero);

    return std::make_pair(invMat.asDiagonal(), numNonZero);
}

auto eigenFactorize(const Eigen::MatrixXd& A, double eps=EPS) {
    // Eigenfactorization of a gram matrix (PSD)

    // NOTE: right most vectors are the eigenvectors with largest eigenvalues
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    // es.compute(A);
    // Eigen::MatrixXd eigvals = es.eigenvalues();
    // Eigen::MatrixXd eigvecs = es.eigenvectors();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    int rank = svd.rank();
    std::cout << "Rank: " << svd.rank() << std::endl;
    Eigen::VectorXd D = svd.singularValues();

    int r = 0;
    for (r = 0; r < rank && D(r) >= eps; ++r);

    D = svd.singularValues().head(r);
    Eigen::MatrixXd U = svd.matrixU().leftCols(r); //.cwiseAbs();
    Eigen::MatrixXd V = svd.matrixV().leftCols(r); //.cwiseAbs();

    std::cout << "U shape: " << U.rows() << " x " << U.cols() << std::endl;
    std::cout << "V shape: " << V.rows() << " x " << V.cols() << std::endl;
    std::cout << "D shape: " << D.rows() << " x " << D.cols() << std::endl;
    for (int j = 0; j < D.cols(); j++) {
        for (int i = 0; i < D.rows(); i++) {
            if (D(i, j) < 0) {
                std::cout << "-ve singular value: " << D(i, j) << " at " << i << ", " << j << std::endl;
            }
        }
    }

    // Eigen::MatrixXd I = V.transpose() * U;
    // assert(I.isIdentity(eps));
    // std::cout << "I: " << std::endl;
    // std::cout << I.topLeftCorner(5, 5) << std::endl;

    return std::make_pair(U, D);
}

auto nystromApproximation(const Eigen::MatrixXd& Ka, const Eigen::MatrixXd& Kab,
                          double eps = EPS)
{

    // Eigendecomposition of Ka
    // Eigen::EigenSolver<Eigen::MatrixXd> es(Ka);
    // // TODO: Check this. This results in a conversion to MatrixXd?
    // Eigen::MatrixXd eigvals = es.eigenvalues().real();
    // Eigen::MatrixXd eigvecs = es.eigenvectors().real();

    // NOTE: right most vectors are the eigenvectors with largest eigenvalues
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    // es.compute(Ka);
    // Eigen::MatrixXd eigvals = es.eigenvalues();
    // Eigen::MatrixXd eigvecs = es.eigenvectors();

    Eigen::MatrixXd eigvals;
    Eigen::MatrixXd eigvecs;
    std::tie(eigvecs, eigvals) = eigenFactorize(Ka);

    // Approximate eigenvectors of K from the above eigenvalues and eigenvectors and Kab
    std::cout << "eigvals shape: " << eigvals.rows() << " x " << eigvals.cols() << std::endl;
    std::cout << "eigvecs shape: " << eigvecs.rows() << " x " << eigvecs.cols() << std::endl;

    int numNonZero;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> invEigVals;
    std::tie(invEigVals, numNonZero) = invertDiagMatrix(eigvals, eps);
    std::cout << "# non-zero eigenvalues: " << numNonZero << std::endl;

    int p = eigvals.rows();
    int n = Ka.cols() + Kab.cols();
    Eigen::MatrixXd phi(n, p);
    phi << eigvecs, (Kab.transpose() * eigvecs * invEigVals);
    return std::make_pair(eigvals, phi);
}

void plotSampledPoints(cv::Mat& I, int nSamples)
{
    int nrows = I.rows;
    int ncols = I.cols;
    std::vector<int> selected, rest;
    std::tie(selected, rest) = samplePixels(nrows, ncols, nSamples);
    for (int i : selected) {
        int r, c;
        std::tie(r, c) = to2DCoords(i, ncols);
        cv::circle(I, cv::Point(c, r), 2, cv::Scalar(255, 0, 0), -1);
    }

    std::cout << "# selected: " << selected.size() << std::endl;
}

double getElement(const Eigen::VectorXd& v, int index) {
    return v(index);
}


cv::Mat eigen2opencv(Eigen::VectorXd& v, int nrows, int ncols) {
    cv::Mat X(nrows, ncols, CV_64FC1, v.data());
    return X;
}

Eigen::VectorXd backwardPermute(const Eigen::VectorXd& v, std::vector<int>& order)
{
    Eigen::VectorXd sortedVec(v.size());

    assert(v.size() == order.size());

    for (int i = 0; i < v.size(); i++) {
        sortedVec(order[i]) = v(i);
    }

    return sortedVec;
}

cv::Mat rescaleForVisualization(const cv::Mat& mat) {
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Mat rescaledMat = (mat - minVal) / (maxVal - minVal) * 255;
    return rescaledMat;
}


auto orthogonalize(Eigen::MatrixXd& Wa, Eigen::MatrixXd& Wab, double eps=EPS) {

    // Eigen::EigenSolver<Eigen::MatrixXd> es(Wa);

    // TODO: Check this. This results in a conversion to MatrixXd?
    // Eigen::MatrixXd eigvals = es.eigenvalues().real();
    // Eigen::MatrixXd eigvecs = es.eigenvectors().real();

    Eigen::MatrixXd eigvals, eigvecs;
    std::tie(eigvecs, eigvals) = eigenFactorize(Wa, eps);

    // Eigen::MatrixXd invRootEigVals = eigvals.cwiseSqrt();
    Eigen::MatrixXd invRootEigVals = eigvals;
    reciprocal(invRootEigVals, eps);
    invRootEigVals = invRootEigVals.cwiseSqrt();

    Eigen::MatrixXd invRootWa = eigvecs * invRootEigVals.asDiagonal() * eigvecs.transpose();

    // TODO: Q is suppose to be symmetric because Wa is suppose to be symmetric
    Eigen::MatrixXd Q = Wa + invRootWa * Wab * Wab.transpose() * invRootWa;

    Eigen::MatrixXd Sq, Vq;
    std::tie(Vq, Sq) = eigenFactorize(Q, eps);

    // Eigen::MatrixXd invRootSq = Sq.cwiseSqrt();
    Eigen::MatrixXd invRootSq = Sq;
    reciprocal(invRootSq, eps);
    invRootSq = invRootSq.cwiseSqrt();

    Eigen::MatrixXd tmp(Wa.rows() + Wab.cols(), Wa.cols());
    tmp << Wa, Wab.transpose();
    Eigen::MatrixXd V = tmp * invRootWa * Vq * invRootSq.asDiagonal();

    if (Q.isApprox(Q.transpose(), eps)) {
        std::cout << "Q is symmetric" << std::endl;
    }
    else {
        std::cout << "Q is NOT symmetric" << std::endl;
    }

    // assert(V.cols() == Sq.rows());

    return std::make_pair(V, Sq);
}

template <typename T>
cv::Mat filterImage(const cv::Mat& I, std::vector<T>& weights)
{
    cv::Mat Ilab;
    cv::cvtColor(I, Ilab, cv::COLOR_BGR2Lab);
    // cv::cvtColor(I, Ilab, cv::COLOR_BGR2YUV);
    // cv::cvtColor(I, Ilab, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(Ilab, channels);

    // TODO: Remember to convert back to 8U before merging
    // TODO: Check if value of L depends on original type.
    cv::Mat L = channels[0];
    L.convertTo(L, CV_64F);

    for (int i = 0; i < L.rows; i++) {
        for (int j = 0; j < L.cols; j++) {
            if (L.at<double>(i, j) < 0) {
                std::cout << "-ve L channel value: " << L.at<double>(i, j) << " at " << i << ", " << j << std::endl;
            }
        }
    }

    // plotSampledPoints(I.clone(), 10);
    // cv::imshow("sampled", I);
    // cv::waitKey(-1);

    std::cout << "Computing kernel weights" << std::endl;
    Eigen::MatrixXd Ka, Kab;
    int nRowSamples = 20;

    // std::vector<int> pixelOrder;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = computeKernelWeights(L, Ka, Kab, nRowSamples);

    Eigen::MatrixXd eigvals, phi;
    std::tie(eigvals, phi) = nystromApproximation(Ka, Kab);

    // Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> d = eigvals.asDiagonal();
    // std::cout << "d shape: " << d.rows() << " x " << d.cols() << std::endl;

    std::cout << "Top 5 Eigenvalue: " << std::endl;
    std::cout << eigvals.topRows(5) << std::endl;
    for (int j = 0; j < eigvals.cols(); j++) {
        for (int i = 0; i < eigvals.rows(); i++) {
            if (eigvals(i, j) < 0) {
                std::cout << i << ", " << j << " -ve eigenvalue: " << eigvals(i, j) << std::endl;
            }
        }
    }

    Eigen::MatrixXd Wa, Wab;
    std::tie(Wa, Wab) = sinkhornKnopp(phi, eigvals, 20);
    if (Wa.isApprox(Wa.transpose()), EPS) {
        std::cout << "Wa is symmetric" << std::endl;
    }
    else {
        std::cout << "Wa is NOT symmetric" << std::endl;
        std::cout << Wa.topLeftCorner(5, 5) << std::endl;
    }

    Eigen::MatrixXd tmp(Wa.rows(), Wa.cols() + Wab.cols());
    tmp << Wa, Wab;
    if (tmp.rowwise().sum().isApprox(Eigen::VectorXd::Ones(Wa.rows()))) {
        std::cout << "Waab is row stochastic" << std::endl;
    }
    else {
        std::cout << "Waab is NOT row stochastic" << std::endl;
    }

    std::cout << "Orthogonalize" << std::endl;
    Eigen::MatrixXd V, S;
    std::tie(V, S) = orthogonalize(Wa, Wab);

    // std::cout << "S top k of total length: " << S.rows() << std::endl;
    // std::cout << S.topRows(7) << std::endl;
    std::cout << "Eigenvalues S" << std::endl;
    std::cout << S << std::endl;

    // Permute eigenvector entries back to the same order as the original image.
    V = P * V;
    const int K = V.cols();
    for (int i = 0; i < V.cols() && i < K; i++) {
        Eigen::VectorXd v = V.col(i);
        std::cout << "eigenvector " << i << " min: " << v.minCoeff() << " max: " << v.maxCoeff() << std::endl;
        cv::Mat ev = eigen2opencv(v, L.rows, L.cols);
        ev = rescaleForVisualization(ev);
        ev.convertTo(ev, CV_8U);
        cv::imshow("ev" + std::to_string(i), ev);
    }

    if ((V.transpose() * V).isIdentity(EPS)) {
        std::cout << "V is orthogonal" << std::endl;
    }
    else {
        std::cout << "V is not orthogonal" << std::endl;
        // std::cout << "V^T * V" << std::endl;
        // std::cout << V.transpose() * V << std::endl;
    }

    std::cout << "Row sum" << std::endl;
    std::cout << (V.row(0) * S.asDiagonal() * V.transpose()).sum() << std::endl;
    std::cout << (V.row(1) * S.asDiagonal() * V.transpose()).sum() << std::endl;
    std::cout << (V.row(2) * S.asDiagonal() * V.transpose()).sum() << std::endl;
    std::cout << (V.row(3) * S.asDiagonal() * V.transpose()).sum() << std::endl;

    cv::waitKey(-1);

    // cv::Mat Lv = L.reshape(0, V.rows());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lv;
    lv.resize(V.rows(), 1);
    int k = 0;
    for (int i = 0; i < L.rows; i++) {
        for (int j = 0; j < L.cols; j++) {
            lv(k) = L.at<double>(i, j);
            ++k;
        }
    }

    std::cout << "Original image " << " min: " << lv.minCoeff() << " max: " << lv.maxCoeff() << std::endl;

    Eigen::VectorXd result = V * (S.asDiagonal() * V.transpose() * lv);
    std::cout << "edited image " << " min: " << result.minCoeff() << " max: " << result.maxCoeff() << std::endl;
    cv::Mat edited = eigen2opencv(result, L.rows, L.cols);
    // edited = rescaleForVisualization(edited);
    edited.convertTo(edited, CV_8U);
    cv::imshow("edited", edited);

    cv::Mat LL;
    L.convertTo(LL, CV_8U);
    cv::imshow("original", LL);
    cv::waitKey(-1);


    // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> intensityEmat(L.ptr<double>(), L.rows, L.cols);
    // intensityEmat.resize(V.rows(), 1);

    // Eigen::Map<MatrixXf, Eigen::Rowwise> intensity( L.data() ); 


    return cv::Mat();
}
