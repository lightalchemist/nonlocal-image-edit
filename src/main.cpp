#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "filter.hpp"


int main(int argc, char* argv[])
{
    if (argc < 13) {
        std::cerr << "Usage: " << argv[0] << " <image> <output> <# row samples> <# col samples> <hx> <hy> <# sinkhorn iterations> <# eigen vectors> <weight 1> <weight 2> <weight 3> <weight 4>" << std::endl;
        return 0;
    }

    std::string imagePath { argv[1] };
    std::string outputPath { argv[2] };
    int nRowSamples = std::stoi(argv[3]);
    int nColSamples = std::stoi(argv[4]);
    double hx = std::stod(argv[5]);
    double hy = std::stod(argv[6]);
    int nSinkhornIter = std::stoi(argv[7]);
    int nEigenVectors = std::stoi(argv[8]);
    std::vector<double> weights;
    for (auto i = 9u; i < argc; ++i) {
        weights.push_back(std::stod(argv[i]));
    }

    std::cout << "Weights: " << std::endl;
    for (auto i : weights) {
        std::cout << i << std::endl;
    }

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to read file from " << imagePath << std::endl;
        return 0;
    }

//    cv::imshow("original", image);
//    cv::waitKey(0);

    // Always convert image to landscape mode first by rotating it
    // Then rotate it back later. This is to ensure we always sample the shorter side
    // and hence less likely to lose information in the image.

    // cv::Mat result = filterImage(image, weights, nRowSamples, nColSamples, hx, hy, nSinkhornIter);
    // cv::imwrite(outputPath, result);
    // cv::Mat colorCorrected = filterImageColorCast(image, weights, nRowSamples, nColSamples, hx, hy, nSinkhornIter);
    // cv::imwrite(outputPath, colorCorrected);
    // cv::imshow("result", colorCorrected);

    cv::Mat result = filterImage2(image, weights, nRowSamples, nColSamples, hx, hy, nSinkhornIter, nEigenVectors);
    cv::imwrite(outputPath, result);
    cv::imshow("result", result);
    cv::waitKey(-1);

    return 0;
}
