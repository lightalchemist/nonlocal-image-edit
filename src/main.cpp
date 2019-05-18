#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "filter.hpp"


int main(int argc, char* argv[])
{
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0] << " <image> <output> <# row samples> <# col samples> <hy> <weight 1> <weight 2> <weight 3> <weight 4>" << std::endl;
        return 0;
    }

    std::string imagePath { argv[1] };
    std::string outputPath { argv[2] };
    int nRowSamples = std::stoi(argv[3]);
    int nColSamples = std::stoi(argv[4]);
    double hy = std::stod(argv[5]);
    std::vector<double> weights;
    for (auto i = 0u; i < 4; ++i) {
        weights.push_back(std::stod(argv[6 + i]));
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

    cv::Mat result = filterImage(image, weights, nRowSamples, nColSamples, hy);
    // cv::imwrite(outputPath, result);

    return 0;
}
