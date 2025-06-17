#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Clase para cargar imágenes desde carpetas
class ImageLoader {
public:
    static std::vector<std::vector<double>> loadImagesFromFolder(const std::string& baseDir) {
        std::vector<std::vector<double>> data;
        for (int i = 0; i <= 7; ++i) {
            std::string classDir = baseDir + "/" + std::to_string(i);
            for (const auto& entry : fs::directory_iterator(classDir)) {
                cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                std::vector<double> vec;
                for (int r = 0; r < img.rows; ++r) {
                    for (int c = 0; c < img.cols; ++c) {
                        vec.push_back(static_cast<double>(img.at<uint8_t>(r, c)) / 255.0);
                    }
                }
                data.push_back(vec);
            }
        }
        return data;
    }
};

// Neurona en el mapa
class Neuron {
public:
    std::vector<double> weights;
    int x, y;

    Neuron(int inputDim, int x_pos, int y_pos) : x(x_pos), y(y_pos) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        weights.resize(inputDim);
        for (int i = 0; i < inputDim; ++i)
            weights[i] = dis(gen);
    }

    double euclideanDistance(const std::vector<double>& input) const {
        double dist = 0.0;
        for (size_t i = 0; i < input.size(); ++i)
            dist += (weights[i] - input[i]) * (weights[i] - input[i]);
        return std::sqrt(dist);
    }
};

// Mapa autoorganizado (SOM)
class SOM {
private:
    std::vector<std::vector<Neuron>> grid;
    int width, height;
    double learningRate;
    double radius;

    double gaussian(double distance, double sigma) {
        return std::exp(-distance * distance / (2.0 * sigma * sigma));
    }

public:
    SOM(int width, int height, int inputDim, double learningRate = 0.1, double radius = 3.0)
        : width(width), height(height), learningRate(learningRate), radius(radius) {
        for (int y = 0; y < height; ++y) {
            std::vector<Neuron> row;
            for (int x = 0; x < width; ++x) {
                row.emplace_back(inputDim, x, y);
            }
            grid.push_back(row);
        }
    }

    void train(const std::vector<std::vector<double>>& data, int epochs) {
        int inputDim = data[0].size();
        double timeConstant = epochs / std::log(radius);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double currentRadius = radius * std::exp(-epoch / timeConstant);
            double currentLearningRate = learningRate * std::exp(-epoch / timeConstant);

            for (const auto& input : data) {
                // Encontrar la mejor neurona (BMU)
                int bmuX = 0, bmuY = 0;
                double minDist = grid[0][0].euclideanDistance(input);
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        double dist = grid[y][x].euclideanDistance(input);
                        if (dist < minDist) {
                            minDist = dist;
                            bmuX = x;
                            bmuY = y;
                        }
                    }
                }

                // Actualizar los pesos de las neuronas cercanas
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        double distToBMU = std::sqrt((x - bmuX) * (x - bmuX) + (y - bmuY) * (y - bmuY));
                        if (distToBMU <= currentRadius) {
                            double influence = gaussian(distToBMU, currentRadius);
                            for (int i = 0; i < inputDim; ++i)
                                grid[y][x].weights[i] += influence * currentLearningRate * (input[i] - grid[y][x].weights[i]);
                        }
                    }
                }
            }
            std::cout << "Epoch " << epoch + 1 << " de " << epochs << " completado.\n";
        }
    }

    // Método opcional para visualizar el mapa (ejemplo básico)
    void visualize() const {
        cv::Mat output(height * 28, width * 28, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                cv::Mat neuronImg(28, 28, CV_8UC1);
                for (int i = 0; i < 28 * 28; ++i) {
                    int row = i / 28, col = i % 28;
                    neuronImg.at<uint8_t>(row, col) = static_cast<uint8_t>(grid[y][x].weights[i] * 255);
                }
                neuronImg.copyTo(output(cv::Rect(x * 28, y * 28, 28, 28)));
            }
        }
        cv::imshow("SOM", output);
        cv::waitKey(0);
    }
};

int main() {
    std::string trainDir = "resources/train";  // Cambia por tu ruta real
    std::cout << "se leyo correctamente la dataset" << std::endl;
    // Cargar datos de entrenamiento
    std::vector<std::vector<double>> data = ImageLoader::loadImagesFromFolder(trainDir);
    std::cout << "Datos cargados: " << data.size() << " imágenes.\n";

    // Crear y entrenar el SOM (10x10 neuronas, entrada de 28x28 = 784 dimensiones)
    SOM som(10, 10, 28 * 28, 0.5, 4.0);
    som.train(data, 100);  // 100 épocas

    // Opcional: visualizar el mapa
    som.visualize();

    return 0;
}
