#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <sutil/Aabb.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <ostream>

// forwrad declarations
class Context;
class HairProgramGroups;


class Sphere {
    public:
        Sphere( const OptixDeviceContext context, const std::string& fileName );
        virtual ~Sphere();

        std::vector<float3> points() const;
        std::vector<float>  radius() const;
        std::vector<float>  value() const;
        std::pair<int,int> printDim() const;
        void           printPoints() const;
        /**
        virtual void gatherProgramGroups( HairProgramGroups* pProgramGroups ) const;

        std::string programName() const;
        std::string programSuffix() const;*/

        sutil::Aabb  aabb() const { return m_aabb; }

    protected:
        OptixTraversableHandle gas() const;

        float defaultRadius() const;

        void makeOptix() const;
        void clearOptix();

    private:
        //TODO: FileHeader          m_header;
        std::vector<float3> m_points;
        std::vector<float>  m_radius;
        std::vector<float>  m_value;
        int m_row;
        int m_col;

        mutable sutil::Aabb m_aabb;

        OptixDeviceContext m_context = 0;

        friend std::ostream& operator<<( std::ostream& o, const Sphere& sphere );
};

// Output operator for Sphere
std::ostream& operator<<( std::ostream& o, const Sphere& sphere );