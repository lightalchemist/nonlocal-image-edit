#include "filter.hpp"

using nle::NLEFilter;
using nle::Vec;
using nle::Mat;
using nle::Point;
using nle::DType;
using nle::OPENCV_MAT_TYPE;
using nle::EPS;


#include <opencv2/highgui.hpp>

inline int to1DIndex(int row, int col, int ncols)
{
    return row * ncols + col;
}

inline std::pair<int, int> to2DCoords(int index, int ncols)
{
    return std::make_pair(index / ncols, index % ncols);
}

cv::Mat eigen2opencv(Vec& v, int nrows, int ncols)
{
    cv::Mat X(nrows, ncols, OPENCV_MAT_TYPE, v.data());
    return X;
}

template <typename T>
Vec opencv2Eigen(const cv::Mat& mat)
{
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

cv::Mat rescaleForVisualization(const cv::Mat& mat)
{
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
            if ((r >= rowOffset && c >= colOffset) && 
                ((r - rowOffset) % rowStep == 0) && 
                ((c - colOffset) % colStep == 0) && 
                r <= (nrows - rowOffset) && c <= (ncols - colOffset)) {
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

Vec transformEigenValues(const Vec& eigvals, const std::vector<DType>& weights)
{
    int nEigVals = eigvals.rows();
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

int robustInplaceReciprocal(Vec& v, DType eps = EPS)
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

#ifdef USE_SPECTRA
auto topkEigenDecomposition(const Mat& M, int nLargest, DType eps=EPS)
{
    nLargest = std::min(nLargest, static_cast<int>(M.rows()));
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
    for (r = 0; r < eigvals.size() && eigvals(r) >= eps; ++r);
    if (r < eigvals.size()) {
        Mat evecs = eigvecs.leftCols(r);
        Vec evals = eigvals.head(r);
        return std::make_pair(evecs, evals);
    }
    else {
        return std::make_pair(eigvecs, eigvals);
    }

}
#endif

// TODO: Implement move semantic version of this to improve
// memory performance of orthogonalize
auto eigenDecomposition(const Mat& M, DType eps=EPS)
{
    // Compute eigen factorization of a PSD matrix
    Eigen::JacobiSVD<Mat> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Vec D = svd.singularValues();
    int rank = svd.rank();
    int r = 0;
    for (r = 0; r < rank && D(r) >= eps; ++r);
    D = (svd.singularValues().head(r)).eval();
    Mat U = svd.matrixU().leftCols(r);
    Mat V = svd.matrixV().leftCols(r);

#ifndef NDEBUG
    // TODO: Replace this with proper logger
    std::cout << "Rank: " << svd.rank() << std::endl;
    std::cout << "U shape: " << U.rows() << " x " << U.cols() << std::endl;
    std::cout << "V shape: " << V.rows() << " x " << V.cols() << std::endl;
    std::cout << "D shape: " << D.rows() << " x " << D.cols() << std::endl;
    std::cout << "Smallest k eigenvalues" << std::endl;
    int k = std::min(5, r);
    std::cout << D.tail(k) << std::endl;
    // Mat mat = V.transpose() * U;
    // assert(mat.isIdentity(eps));
    // std::cout << "mat: " << std::endl;
    // std::cout << mat.topLeftCorner(5, 5) << std::endl;
#endif

    return std::make_pair(U, D);
}

cv::Mat NLEFilter::denoise(const cv::Mat& image, DType k) const
{
    std::cout << "k: " << k << std::endl;

    if (image.channels() != 3) {
        throw std::runtime_error("Can only enchance RGB image.");
    }

    if (image.total() != m_eigvecs.rows()) {
        throw std::runtime_error("Cannot apply filter on image with different size from the image filter was trained on.");
    }

    cv::Mat II;
    cv::cvtColor(image, II, cv::COLOR_BGR2YUV);
    // cv::cvtColor(image, II, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(II, channels);
    channels[0].convertTo(channels[0], OPENCV_MAT_TYPE);
    channels[1].convertTo(channels[1], OPENCV_MAT_TYPE);
    channels[2].convertTo(channels[2], OPENCV_MAT_TYPE);

    Vec teigvals = Vec(m_eigvals.size());
    for (int i = 0; i < teigvals.size(); i++) {
        DType eval = teigvals(i);
        std::cout << "eig " << i << " val: " << eval << std::endl;
        teigvals(i) = std::pow(eval, k);
    }

    channels[1] = apply(channels[1], teigvals);
    channels[2] = apply(channels[2], teigvals);

    // TODO: Check if we can do this inplace
    channels[0].convertTo(channels[0], CV_8U);
    // channels[1] = cv::max(channels[1], 0);
    // channels[1] = cv::min(channels[1], 255);
    channels[1].convertTo(channels[1], CV_8U);
    // channels[2] = cv::max(channels[2], 0);
    // channels[2] = cv::min(channels[2], 255);
    channels[2].convertTo(channels[2], CV_8U);

    cv::imshow("luminosity channel", channels[0]);

    cv::Mat filteredImage;
    cv::merge(channels, filteredImage);
    cv::cvtColor(filteredImage, filteredImage, cv::COLOR_YUV2BGR);
    // cv::cvtColor(filteredImage, filteredImage, cv::COLOR_Lab2BGR);

    return filteredImage;
}

cv::Mat NLEFilter::enhance(const cv::Mat& image, const std::vector<DType>& weights) const
{
    if (image.channels() != 3) {
        throw std::runtime_error("Can only enchance RGB image.");
    }

    if (image.total() != m_eigvecs.rows()) {
        throw std::runtime_error("Cannot apply filter on image with different size from the image filter was trained on.");
    }

    cv::Mat II;
    cv::cvtColor(image, II, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(II, channels);
    channels[0].convertTo(channels[0], OPENCV_MAT_TYPE);

    int k = std::min(10, static_cast<int>(m_eigvals.size()));
    std::cout << "Original eigvals: " << std::endl;
    std::cout << m_eigvals.head(k) << std::endl;
    
    Vec fS = transformEigenValues(m_eigvals, weights);
    std::cout << "Transformed eigvals fS: " << std::endl << fS.head(k) << std::endl;
    
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

    Vec c = opencv2Eigen<DType>(channel);
    Vec filtered = m_eigvecs * (transformedEigVals.asDiagonal() * m_eigvecs.transpose() * c);
    return eigen2opencv(filtered, channel.rows, channel.cols);
}

auto NLEFilter::computeKernelWeights(const cv::Mat& mat, int nRowSamples, int nColSamples,
                                     DType hx, DType hy) const
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

    // TODO: Check if Eigen aliasing can cause problem here.
    // Convert Ka and Kab to kernels
    // Ka = Ka.array().exp();
    // Kab = Kab.array().exp();
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

std::pair<Mat, Mat>
NLEFilter::sinkhornKnopp(const Mat& phi, const Vec& eigvals, int maxIter, DType eps) const
{
    int n = phi.rows();
    Vec r = Vec::Ones(n, 1);
    Vec c;

    Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic> D = eigvals.asDiagonal();
    for (int i = 0; i < maxIter; i++) {
        c = phi * (D * (phi.transpose() * r));
        robustInplaceReciprocal(c, eps);
        assert(c.rows() == phi.rows());
        assert(c.cols() == 1);
        r = phi * (D * (phi.transpose() * c));
        robustInplaceReciprocal(r, eps);
    }

    int p = phi.cols();
    Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic> R = r.head(p).asDiagonal();
    Mat Wa = (R * (phi.topRows(p) * D)) * (c.head(p).replicate(1, p).array() * phi.topRows(p).array()).matrix().transpose();
    Mat Wab = (R * (phi.topRows(p) * D)) * (c.tail(n - p).replicate(1, p).array() * phi.bottomRows(n - p).array()).matrix().transpose();

    assert(Wa.cols() + Wab.cols() == phi.rows());
    return std::make_pair(Wa, Wab);
}

// TODO: Implement move semantic version of this to improve memory performance
auto NLEFilter::nystromApproximation(const Mat& Ka, const Mat& Kab, DType eps) const
{
    Vec eigvals;
    Mat eigvecs;
    std::tie(eigvecs, eigvals) = eigenDecomposition(Ka);

    // Approximate eigenvectors of K from the above eigenvalues and eigenvectors and Kab
    Vec tmp = eigvals;
    int numNonZero = robustInplaceReciprocal(tmp);
    Eigen::DiagonalMatrix<DType, Eigen::Dynamic, Eigen::Dynamic> invEigVals = tmp.head(numNonZero).asDiagonal();
    eigvecs = eigvecs.leftCols(numNonZero).eval();

    int n = Ka.cols() + Kab.cols();
    Mat phi(n, eigvecs.cols());
    phi << eigvecs, (Kab.transpose() * eigvecs * invEigVals);
    return std::make_pair(eigvals, phi);
}

auto NLEFilter::orthogonalize(const Mat& Wa, const Mat& Wab, int nEigVectors, DType eps) const
{
    Mat eigvecs;
    Vec eigvals;
    std::tie(eigvecs, eigvals) = eigenDecomposition(Wa, eps);

    Vec invRootEigVals = eigvals;
    robustInplaceReciprocal(invRootEigVals, eps);
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
    std::tie(Vq, Sq) = topkEigenDecomposition(Q, nEigVectors, eps);
#else
    std::tie(Vq, Sq) = eigenDecomposition(Q, eps);
    
    std::cout << "Original # eigenvalues: " << Sq.rows() << " x " << Sq.cols() << std::endl;
    
    int k = std::min(nEigVectors, static_cast<int>(Vq.cols()));
    Vq = Vq.leftCols(k).eval();
    Sq = Sq.head(k).eval();
#endif

    Vec invRootSq = Sq;
    robustInplaceReciprocal(invRootSq, eps);
    invRootSq = invRootSq.cwiseSqrt();

    // Stack Wa above Wab^T
    Mat tmp(Wa.rows() + Wab.cols(), Wa.cols());
    tmp << Wa, Wab.transpose();
    // Compute approximate eigenvectors of W
    Mat V = tmp * invRootWa * Vq * invRootSq.asDiagonal();

    // V contain eigenvectors of W and Sq contains their corresponding eigenvalues
    return std::make_pair(V, Sq);
}

cv::Mat getLuminosityChannel(const cv::Mat& image)
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
    channels[0].convertTo(channels[0], OPENCV_MAT_TYPE);
    cv::Mat Y = channels[0];
    return Y;
}

void
NLEFilter::learnForEnhancement(const cv::Mat& image, int nRowSamples, int nColSamples,
                               DType hx, DType hy, int nSinkhornIter, int nEigenVectors)
{
    cv::Mat luminosity = getLuminosityChannel(image);

    std::cout << "Computing kernel" << std::endl;
    Mat Ka, Kab;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P;
    std::tie(P, Ka, Kab) = computeKernelWeights(luminosity, nRowSamples, nColSamples, hx, hy);

    std::cout << "Nystrom approximation" << std::endl;
    Vec eigvals;
    Mat phi;
    std::tie(eigvals, phi) = nystromApproximation(Ka, Kab);

    std::cout << "Sinkhorn" << std::endl;
    Mat Wa, Wab;
    std::tie(Wa, Wab) = sinkhornKnopp(phi, eigvals, nSinkhornIter);
     Wa = (Wa + Wa.transpose()).eval() / 2;

    std::cout << "Orthogonalize" << std::endl;
//    Mat V;
//    Vec S;
//    std::tie(V, S) = orthogonalize(Wa, Wab, nEigenVectors);
//    int nFilters = std::min(nEigenVectors, static_cast<int>(S.rows()));
//    m_eigvecs = V.leftCols(nFilters).eval();
//    m_eigvals = S.head(nFilters).eval();
    
    std::tie(m_eigvecs, m_eigvals) = orthogonalize(Wa, Wab, nEigenVectors);
    
    int k = std::min(int(m_eigvals.size()), 5);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            std::cout << "v" << i << " dot " << "v" << j << ": " << m_eigvecs.col(i).dot(m_eigvecs.col(j)) << std::endl;
        }
    }
    

    // Permute values back into correct position
    m_eigvecs = (P * m_eigvecs).eval();

    for (int i = 0; i < std::min(nEigenVectors, 5); i++) {
         Vec v = m_eigvecs.col(i);
        std::cout << "Eigvec " << i << " minCoeff: " << v.minCoeff() << " maxCoeff: " << v.maxCoeff() << std::endl;
         cv::Mat m = eigen2opencv(v, image.rows, image.cols);
         m = rescaleForVisualization(m);
         m.convertTo(m, CV_8U);
         cv::imshow("image" + std::to_string(i), m);
     }
}


void
NLEFilter::learnForDenoise(const cv::Mat& image, int nRowSamples, int nColSamples,
                           DType hx, DType hy, int nSinkhornIter, int nEigenVectors)
{
    cv::Mat Y = getYChannel(image);

    std::cout << "Computing kernel" << std::endl;
    Mat Ka, Kab;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P;
    std::tie(P, Ka, Kab) = computeKernelWeights(Y, nRowSamples, nColSamples, hx, hy);

    std::cout << "Nystrom approximation" << std::endl;
    Vec eigvals;
    Mat phi;
    std::tie(eigvals, phi) = nystromApproximation(Ka, Kab);

    std::cout << "Sinkhorn" << std::endl;
    Mat Wa, Wab;
    std::tie(Wa, Wab) = sinkhornKnopp(phi, eigvals, nSinkhornIter);
    // Wa = (Wa + Wa.transpose()) / 2;

    std::cout << "Orthogonalize" << std::endl;
    Mat V;
    Vec S;
    std::tie(V, S) = orthogonalize(Wa, Wab, nEigenVectors);

    int nFilters = std::min(nEigenVectors, static_cast<int>(S.rows()));
    m_eigvecs = V.leftCols(nFilters).eval();
    m_eigvals = S.head(nFilters).eval();

    // Permute values back into correct position
    m_eigvecs = (P * m_eigvecs).eval();

    // for (int i = 0; i < nEigenVectors; i++) {
    //     Vec v = m_eigvecs.col(i);
    //     cv::Mat m = eigen2opencv(v, image.rows, image.cols);
    //     m = rescaleForVisualization(m);
    //     m.convertTo(m, CV_8U);
    //     cv::imshow("image" + std::to_string(i), m);
    // }
    // cv::waitKey(-1);
}
