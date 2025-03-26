#include "Sphere.h"

#include <sutil/sutil.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <iostream>

#include "Util.h"

// #define MAX_NUM_ROW 55548
#define MAX_NUM_ROW 42548

/**
 *
 * Generate sphere from file
 * Sets up the sphere parameter
 * @param fileName, file containing a subset of matrix data needed to be processed
 *        matrix data in COO format (x, y, value)
 *        stored in Sphere as points = (x, y, 0) and value = (value)
 *
 */
Sphere::Sphere( const OptixDeviceContext context, const std::string& fileName )
    : m_context( context )
{
    
    std::ifstream file(fileName);
    std::string line;

    std::ifstream input( fileName.c_str(), std::ios::in );
    SUTIL_ASSERT_MSG( input.is_open(), "Unable to open " + fileName + "." );

    bool isFirstDataLine = true;
    int rows, cols;
    uint64_t nonZeros;

    bool isLargerThanMem = false;
    int midRowStart, midRowEnd, midColStart, midColEnd;

    while (std::getline(file, line)) {
        // Skip comment lines
        if (line.empty() || line[0] == '%') {
            continue;
        }

        if (isFirstDataLine) {
            // Read the first data line containing dimensions
            std::istringstream iss(line);
            if (!(iss >> rows >> cols >> nonZeros)) {
                std::cerr << "Error reading matrix dimensions." << std::endl;
                return;
            }

            this->m_row = rows;
            this->m_col = cols;
            isFirstDataLine = false;
            continue;
        }

        std::istringstream iss(line);
        float x, y, val;
        if (!(iss >> x >> y >> val)) {
            // Handle parsing error
            continue; 
        }

        float3 point;
        
        point = make_float3( x - 1, y - 1, 0.f ); // convert to 0-based            

        m_points.push_back(point); // Z-coordinate is 0 for 2D data
        m_radius.push_back(0.1f);
        m_value.push_back(val);
    }
}

Sphere::~Sphere() {}

float Sphere::defaultRadius() const
{
    float defaultRadius = 0.1f;
    return defaultRadius;
    //return m_header.defaultRadius;
}

/**
 *
 * @return a list of float3 data that marks points of the sphere
 *         each data should be in the format of (x, y, 0) which marks 
 *         the x and y axis in the matrix of the current value.
 *
 */
std::vector<float3> Sphere::points() const
{
    return m_points;
}


/**
 *
 * @return a list of radius, which are remained the same for every data points
 *
 */
std::vector<float>  Sphere::radius() const
{
    return m_radius;
}

/**
 *
 * @return the list of value of the matrix data
 * correspond to the (x,y) coordinates in points()
 *
 */
std::vector<float>  Sphere::value() const
{
    return m_value;
}

std::pair<int,int> Sphere::printDim() const
{
    return std::make_pair(m_row, m_col);
}

void Sphere::printPoints() const
{
    for( auto point : m_points )
    {
        std::cout << point << std::endl;
    }
    return;
}