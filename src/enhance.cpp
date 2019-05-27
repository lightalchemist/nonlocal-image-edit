#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "filter.hpp"


int main(int argc, char* argv[])
{
    // TODO: Replace with proper argument parser
    if (argc < 10) {
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
    std::vector<nle::DType> weights;
    for (auto i = 9; i < argc; ++i) {
        weights.push_back(std::stod(argv[i]));
    }

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to read file from " << imagePath << std::endl;
        return 0;
    }

    auto filter = nle::NLEFilter();

    // See this link for effect of matrix type on output range of Lab conversion.
    // https://stackoverflow.com/questions/11386556/converting-an-opencv-bgr-8-bit-image-to-cie-lab
    filter.trainForEnhancement(image, nRowSamples, nColSamples, hx, hy, nSinkhornIter, nEigenVectors);
    cv::Mat result = filter.enhance(image, weights);
    std::cout << "Done. Press any key in result window to exit." << std::endl;

    cv::imwrite(outputPath, result);
    cv::imshow("Result", result);
    cv::waitKey(-1);
        
    return 0;
}
