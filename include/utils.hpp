#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

namespace nle {

    inline int to1DIndex(int row, int col, int ncols)
    {
        return row * ncols + col;
    }

    inline std::pair<int, int> to2DCoords(int index, int ncols)
    {
        return std::make_pair(index / ncols, index % ncols);
    }

    static cv::Mat eigen2opencv(Vec& v, int nrows, int ncols)
    {
        cv::Mat X(nrows, ncols, OPENCV_MAT_TYPE, v.data());
        return X.clone();
    }

    template <typename T>
    Vec opencv2eigen(const cv::Mat& mat)
    {
        Vec lv(mat.total());
        int k = 0;
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                lv(k) = mat.at<T>(i, j);
                ++k;
            }
        }

        return lv;
    }
}

#endif
