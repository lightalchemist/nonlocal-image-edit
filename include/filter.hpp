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


// const double EPS = 1e-19;
// const double EPS = 1e-20;
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

double kernel(const cv::Mat& I,
              int r1, int c1, int r2, int c2,
              double spatialScale, double intensityScale)
{
    double yr = I.at<double>(r1, c1);
    double ys = I.at<double>(r2, c2);

    double squareSpatialDist = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2);
    double squareIntensityDist = (yr - ys) * (yr - ys);
    return std::exp(-spatialScale * squareSpatialDist - intensityScale * squareIntensityDist);
}

double kernel(const cv::Mat& I, const Point& p1, const Point& p2,
              double spatialScale, double intensityScale)
{
    double yr = I.at<double>(p1.row, p1.col);
    double ys = I.at<double>(p2.row, p2.col);
    double squareSpatialDist = (p1.row - p2.row) * (p1.row - p2.row) + (p1.col - p2.col) * (p1.col - p2.col);
    double squareIntensityDist = (yr - ys) * (yr - ys);
    return std::exp(-spatialScale * squareSpatialDist - intensityScale * squareIntensityDist);
}

void printNegativeEntries(const Eigen::MatrixXd& mat) {
    for (int j = 0; j < mat.cols(); j++) {
        for (int i = 0; i < mat.rows(); i++) {
            if (mat(i, j) < 0) {
                std::cout << i << ", " << j << " -ve val: " << mat(i, j) << std::endl;
            }
        }
    }
}

auto samplePixels(int nrows, int ncols, int nRowSamples, int nColSamples) 
{
    std::cout << "samplePixels2" << std::endl;

    // int rowStep = nrows / (nRowSamples + 1);
    // int colStep = ncols / (nColSamples + 1);
    int rowStep = nrows / (nRowSamples );
    int colStep = ncols / (nColSamples );

    int rowOffset = (rowStep - 1 + (nrows - rowStep * nRowSamples)) / 2;
    int colOffset = (colStep - 1 + (ncols - colStep * nColSamples)) / 2;

    std::vector<Point> selected, rest;
    selected.reserve(nRowSamples * nColSamples);
    rest.reserve(nrows * ncols - nRowSamples * nColSamples);

    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            if ((r >= rowOffset && c >= colOffset) && ((r - rowOffset) % rowStep == 0) && ((c - colOffset) % colStep == 0) && r <= (nrows - rowOffset) && c <= (ncols - colOffset)) {
                selected.push_back(Point{r, c});
            } else {
                rest.push_back(Point{r, c});
            }
        }
    }

    return std::make_pair(selected, rest);
}

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
computeKernelWeights(const cv::Mat& I, Eigen::MatrixXd& Ka, Eigen::MatrixXd& Kab,
                      int nRowSamples, int nColSamples, double hx, double hy)
{
    int nrows = I.rows, ncols = I.cols;
    int nPixels = I.total();

    std::vector<Point> selected, rest;
    std::tie(selected, rest) = samplePixels(nrows, ncols, nRowSamples, nColSamples);
    int nSamples = selected.size();
    Ka.resize(nSamples, nSamples);
    Kab.resize(nSamples, nPixels - nSamples);

    double gammaIntensity = 1.0 / (hy * hy);
    double gammaSpatial = 1.0 / (hx * hx);

    for (auto i = 0u; i < selected.size(); ++i) {
        // Ka
        Point p1 = selected[i];
        for (auto j = i; j < selected.size(); ++j) {
            auto val = kernel(I, p1, selected[j], gammaSpatial, gammaIntensity);
            Ka(i, j) = val;
            Ka(j, i) = val;
        }

        // Kab
        for (auto j = 0u; j < rest.size(); j++) {
            Kab(i, j) = kernel(I, p1, rest[j], gammaSpatial, gammaIntensity);
        }
    }
    
    // TODO
    
    assert(Ka.isApprox(Ka.transpose()));
    if (Ka.isApprox(Ka.transpose())) {
        std::cout << "Ka is symmetric" << std::endl;
    }
    else {
        std::cout << "Ka is NOT symmetric" << std::endl;
    }

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(nPixels);
    for (int i = 0; i < selected.size(); ++i) {
        Point p = selected[i];
        P.indices()[i] = to1DIndex(p.row, p.col, ncols);
    }
    for (int j = 0; j < rest.size(); ++j) {
        Point p = rest[j];
        P.indices()[j + selected.size()] = to1DIndex(p.row, p.col, ncols);
    }

    return P;
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
//    Eigen::MatrixXd r = Eigen::VectorXd::Ones(n, 1);
    Eigen::VectorXd r = Eigen::VectorXd::Ones(n, 1);
    Eigen::MatrixXd c;
    Eigen::MatrixXd Dphi_t = eigvals.asDiagonal() * phi.transpose();
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

    return std::make_pair(invMat.asDiagonal(), numNonZero);
}

auto eigenDecomposition(const Eigen::MatrixXd& A, double eps=EPS) {
    // Eigenfactorization of a gram matrix (PSD)

    // NOTE: right most vectors are the eigenvectors with largest eigenvalues
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    // es.compute(A);
    // Eigen::MatrixXd eigvals = es.eigenvalues();
    // Eigen::MatrixXd eigvecs = es.eigenvectors();
    
    // Eigendecomposition of Ka
    // Eigen::EigenSolver<Eigen::MatrixXd> es(Ka);
    // // TODO: Check this. This results in a conversion to MatrixXd?
    // Eigen::MatrixXd eigvals = es.eigenvalues().real();
    // Eigen::MatrixXd eigvecs = es.eigenvectors().real();
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    int rank = svd.rank();
    std::cout << "Rank: " << svd.rank() << std::endl;
    Eigen::VectorXd D = svd.singularValues();

    int r = 0;
    for (r = 0; r < rank && D(r) >= eps; ++r);

    D = svd.singularValues().head(r);
    Eigen::MatrixXd U = svd.matrixU().leftCols(r);
    Eigen::MatrixXd V = svd.matrixV().leftCols(r);

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
    Eigen::MatrixXd eigvals;
    Eigen::MatrixXd eigvecs;
    std::tie(eigvecs, eigvals) = eigenDecomposition(Ka);

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

void plotSampledPoints2(cv::Mat& I, int nRowSamples, int nColSamples)
{
    std::cout << "plotSampledPoints2" << std::endl;

    int nrows = I.rows;
    int ncols = I.cols;
    std::vector<Point> selected, rest;
    std::tie(selected, rest) = samplePixels(nrows, ncols, nRowSamples, nColSamples);
    for (Point p : selected) {
        int r = p.row, c = p.col;
        cv::circle(I, cv::Point(c, r), 2, cv::Scalar(255, 0, 0), -1);
    }

    std::cout << "# row samples: " << nRowSamples << " # col samples: " << nColSamples <<  " # selected: " << selected.size() << " # rest: " << rest.size() << std::endl;
}

cv::Mat eigen2opencv(Eigen::VectorXd& v, int nrows, int ncols) {
    cv::Mat X(nrows, ncols, CV_64FC1, v.data());
    return X;
}

template <typename T>
Eigen::VectorXd opencv2Eigen(const cv::Mat& I) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> lv;
    lv.resize(I.total(), 1);
    int k = 0;
    for (int i = 0; i < I.rows; i++) {
        for (int j = 0; j < I.cols; j++) {
            lv(k) = I.at<T>(i, j);
            ++k;
        }
    }
    
    return lv;
}


cv::Mat rescaleForVisualization(const cv::Mat& mat) {
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Mat rescaledMat = (mat - minVal) / (maxVal - minVal) * 255;
    return rescaledMat;
}


auto orthogonalize(Eigen::MatrixXd& Wa, Eigen::MatrixXd& Wab, double eps=EPS) {
    Eigen::MatrixXd eigvals, eigvecs;
    std::tie(eigvecs, eigvals) = eigenDecomposition(Wa, eps);

    Eigen::MatrixXd invRootEigVals = eigvals;
    reciprocal(invRootEigVals, eps);
    invRootEigVals = invRootEigVals.cwiseSqrt();

    Eigen::MatrixXd invRootWa = eigvecs * invRootEigVals.asDiagonal() * eigvecs.transpose();
    Eigen::MatrixXd Q = Wa + invRootWa * Wab * Wab.transpose() * invRootWa;
    if (Q.isApprox(Q.transpose(), eps)) {
        std::cout << "Q is symmetric" << std::endl;
    }
    else {
        std::cout << "Q is NOT symmetric" << std::endl;
        std::cout << Q.bottomRightCorner(5, 5) << std::endl;
    }

    Eigen::MatrixXd Sq, Vq;
    std::tie(Vq, Sq) = eigenDecomposition(Q, eps);

    Eigen::MatrixXd invRootSq = Sq;
    reciprocal(invRootSq, eps);
    invRootSq = invRootSq.cwiseSqrt();

    Eigen::MatrixXd tmp(Wa.rows() + Wab.cols(), Wa.cols());
    tmp << Wa, Wab.transpose();
    Eigen::MatrixXd V = tmp * invRootWa * Vq * invRootSq.asDiagonal();

    // assert(V.cols() == Sq.rows());

    return std::make_pair(V, Sq);
}

Eigen::MatrixXd transformEigenValues(const Eigen::MatrixXd& eigvals,
                                     const std::vector<double>& weights) {
    int nEigVals = eigvals.rows();
    Eigen::MatrixXd fS(nEigVals, 1);
    for (int i = 0; i < nEigVals; i++) {
        double eig = eigvals(i, 0);
        fS(i) = weights[0];
        for (auto k = 1ul; k < weights.size(); k++) {
            fS(i) += (weights[k] - weights[k - 1]) * std::pow(eig, double(k));
        }
    }

    return fS;
}


template <typename T>
std::pair<Eigen::MatrixXd, Eigen::DiagonalMatrix<T, Eigen::Dynamic, Eigen::Dynamic> > 
learnFilter(const cv::Mat& I, std::vector<T>& weights, int nRowSamples, int nColSamples, double hx, double hy, int nSinkhornIter, int nEigenVectors=20)
{
    std::cout << "Computing kernel weights" << std::endl;
    Eigen::MatrixXd Ka, Kab;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = computeKernelWeights(I, Ka, Kab, nRowSamples, nColSamples, hx, hy);

    std::cout << "Running Nystrom approximation" << std::endl;
    Eigen::MatrixXd eigvals, phi;
    std::tie(eigvals, phi) = nystromApproximation(Ka, Kab);

    std::cout << "Top 5 Eigenvalue: " << std::endl;
    std::cout << eigvals.topRows(5) << std::endl;

    std::cout << "Running Sinkhorn Knopp algorithm." << std::endl;
    Eigen::MatrixXd Wa, Wab;
    std::tie(Wa, Wab) = sinkhornKnopp(phi, eigvals, nSinkhornIter);

    std::cout << "Orthogonalize" << std::endl;
    Eigen::MatrixXd V, S;
    std::tie(V, S) = orthogonalize(Wa, Wab);

    std::cout << "S top k of total length: " << S.rows() << std::endl;
    int nEigVals = S.rows();
    int t = std::min(nEigVals, nEigenVectors);
    std::cout << S.topRows(t) << std::endl;
    std::cout << "S bottom k" << std::endl;
    std::cout << S.bottomRows(5) << std::endl;

    V = P * V;
    Eigen::MatrixXd fS = transformEigenValues(S, weights);
    std::cout << "Filtered fS top k" << std::endl;
    std::cout << fS.topRows(10) << std::endl;
    std::cout << "Filtered fS bottom k" << std::endl;
    std::cout << fS.bottomRows(10) << std::endl;

    int nFilters = std::min(5, (int) fS.rows());
    return std::make_pair(V.leftCols(nFilters), fS.topRows(nFilters).asDiagonal());
}

template <typename T>
cv::Mat denoiseColorImage(const cv::Mat& I, std::vector<T>& weights, int nRowSamples, int nColSamples, double hx, double hy, int nSinkhornIter)
{
    cv::Mat II;

    // cv::cvtColor(I, II, cv::COLOR_BGR2YUV);
     cv::cvtColor(I, II, cv::COLOR_BGR2Lab);
    // cv::cvtColor(I, Ilab, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(II, channels);

    channels[0].convertTo(channels[0], CV_64F);
    channels[1].convertTo(channels[1], CV_64F);
    channels[2].convertTo(channels[2], CV_64F);

    Eigen::MatrixXd V, S;
    std::tie(V, S) = makeFilter(channels[0], weights, nRowSamples, nColSamples, hx, hy, nSinkhornIter);

    Eigen::VectorXd u = opencv2Eigen<double>(channels[1]);
    Eigen::VectorXd v = opencv2Eigen<double>(channels[2]);

    Eigen::VectorXd filteredU = V * (S * V.transpose() * u);
    Eigen::VectorXd filteredV = V * (S * V.transpose() * v);

    cv::Mat matU = eigen2opencv(filteredU, I.rows, I.cols);
    cv::Mat matV = eigen2opencv(filteredV, I.rows, I.cols);
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

template <typename T>
cv::Mat filterImage(const cv::Mat& I, std::vector<T>& weights, int nRowSamples, int nColSamples, double hx, double hy, int nSinkhornIter, int nEigenVectors=20)
{
    cv::Mat II;

    // cv::cvtColor(I, II, cv::COLOR_BGR2YUV);
     cv::cvtColor(I, II, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(II, channels);

    channels[0].convertTo(channels[0], CV_64F);
    // channels[1].convertTo(channels[1], CV_64F);
    // channels[2].convertTo(channels[2], CV_64F);

    Eigen::MatrixXd V;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> S;
    std::tie(V, S) = learnFilter(channels[0], weights, nRowSamples, nColSamples, hx, hy, nSinkhornIter, nEigenVectors);

    Eigen::VectorXd y = opencv2Eigen<double>(channels[0]);
    Eigen::VectorXd filteredY = V * (S * V.transpose() * y);

    cv::Mat matY = eigen2opencv(filteredY, I.rows, I.cols);
    matY.convertTo(matY, CV_8U);

    channels[0] = matY;
    channels[0].convertTo(channels[0], CV_8U);

    cv::Mat filteredImage;
    cv::merge(channels, filteredImage);

    // cv::cvtColor(filteredImage, filteredImage, cv::COLOR_YUV2BGR);
    cv::cvtColor(filteredImage, filteredImage, cv::COLOR_Lab2BGR);
    return filteredImage;
}
