#include <Eigen/Dense>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace Eigen
{
    typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
}

namespace grid_map_raycasting
{
    Eigen::MatrixXd createElevationMap(
        const std::vector<Eigen::Vector3d>& points, 
        float voxel_size, 
        int width, 
        int height, 
        const Eigen::Vector2f& origin = Eigen::Vector2f(0.0, 0.0))
    {
        // Calculate the bounds of the area we want to include
        float half_width = (width * voxel_size) / 2;
        float half_height = (height * voxel_size) / 2;
        Eigen::Vector3f min_bound(origin[0] - half_width, origin[1] - half_height, -std::numeric_limits<float>::infinity());
        Eigen::Vector3f max_bound(origin[0] + half_width, origin[1] + half_height, std::numeric_limits<float>::infinity());

        // Crop the point cloud to the specified bounds
        // auto bbox = geometry::AxisAlignedBoundingBox(min_bound, max_bound);
        // auto cropped_pcd = pcd.Crop(bbox);

        // Create the elevation map
        Eigen::MatrixXd elevation_map = Eigen::MatrixXd::Constant(width, height, std::numeric_limits<float>::quiet_NaN());
        
        for (const auto& point : points) {
            auto x = point(0);
            auto y = point(1);
            auto z = point(2);
            int x_idx = static_cast<int>((x - origin[0]) / voxel_size + width / 2);
            int y_idx = static_cast<int>((y - origin[1]) / voxel_size + height / 2);
            if (0 <= x_idx && x_idx < width && 0 <= y_idx && y_idx < height) {
                if (std::isnan(elevation_map(x_idx, y_idx))) {
                    elevation_map(x_idx, y_idx) = z;
                } else {
                    elevation_map(x_idx, y_idx) = std::max(elevation_map(x_idx, y_idx), z);
                }
            }
        }

        return elevation_map;
    }

    Eigen::MatrixXb rayCastGridMap(Eigen::Vector3d vantage_point, Eigen::MatrixXd grid_map, Eigen::Vector2d grid_resolution)
    {
        // the vantage point needs to lie within grid
        assert(-grid_map.rows() / 2 <= vantage_point(0) / grid_resolution(0) <= grid_map.rows() / 2);
        assert(-grid_map.cols() / 2 <= vantage_point(1) / grid_resolution(1) <= grid_map.cols() / 2);

        Eigen::MatrixXb occlusion_mask(grid_map.rows(), grid_map.cols());
        occlusion_mask.setConstant(false);

        for (int i = 0; i < grid_map.rows(); i++)
        {
            for (int j = 0; j < grid_map.cols(); j++)
            {
                double grid_cell_x = (-grid_map.rows() / 2 + i) * grid_resolution(0);
                double grid_cell_y = (-grid_map.cols() / 2 + j) * grid_resolution(1);
                double grid_cell_z = grid_map(i, j);

                if (std::isnan(grid_cell_z))
                {
                    // we skip already occluded cells to improve computational efficiency
                    occlusion_mask(i, j) = true;
                    continue;
                }

                Eigen::Vector3d grid_cell_pos = {grid_cell_x, grid_cell_y, grid_cell_z};

                Eigen::Vector3d direction = grid_cell_pos - vantage_point;
                double ray_length = direction.norm();
                direction /= ray_length;

                Eigen::Vector3d raycast_pos = vantage_point;
                bool grid_cell_occluded = false;
                while (grid_cell_occluded == false)
                {
                    raycast_pos += 0.5 * direction * std::min(grid_resolution(0), grid_resolution(1));

                    int raycast_u = (int)std::round(grid_map.rows() / 2 + raycast_pos(0) / grid_resolution(0));
                    int raycast_v = (int)std::round(grid_map.cols() / 2 + raycast_pos(1) / grid_resolution(1));

                    // the grid_cell cannot occlude itself, thats why we do not consider the final cell
                    if ((i == raycast_u && j == raycast_v))
                    {
                        // std::cout << "break because we reached grid_cell for i=" << i << ", j=" << j << std::endl;
                        break;
                    }

                    if ((raycast_pos - vantage_point).norm() > ray_length)
                    {
                        // std::cout << "break because we are past max_distance for i=" << i << ", j=" << j << std::endl;
                        break;
                    }

                    double ground_elevation = grid_map(raycast_u, raycast_v);

                    // we do not consider cells with missing elevation information to cause occlusion
                    if (!std::isnan(ground_elevation))
                    {
                        // we consider a cell to be occluded, if the ray hits a higher elevation on its trajectory to the cell
                        if (ground_elevation > raycast_pos(2))
                        {
                            grid_cell_occluded = true;
                            // std::cout << "break because we found occluded grid_cell for u=" << raycast_u << ", v=" << raycast_v << std::endl;
                        }
                    }
                }

                occlusion_mask(i, j) = grid_cell_occluded;
            }
        }

        return occlusion_mask;
    }

    Eigen::MatrixXd rayCastElevationMap(Eigen::Vector3d vantage_point, Eigen::MatrixXd grid_map, Eigen::Vector2d grid_resolution)
    {
        // the vantage point needs to lie within grid
        assert(-grid_map.rows() / 2 <= vantage_point(0) / grid_resolution(0) <= grid_map.rows() / 2);
        assert(-grid_map.cols() / 2 <= vantage_point(1) / grid_resolution(1) <= grid_map.cols() / 2);

        Eigen::MatrixXb occlusion_mask(grid_map.rows(), grid_map.cols());
        occlusion_mask.setConstant(false);

        // In the elevation map, each grid height is less than its value
        Eigen::MatrixXd elevation_map = Eigen::MatrixXd::Constant(grid_map.rows(), grid_map.cols(), std::numeric_limits<double>::max());

        for (int i = 0; i < grid_map.rows(); i++)
        {
            for (int j = 0; j < grid_map.cols(); j++)
            {
                double grid_cell_x = (-grid_map.rows() / 2 + i) * grid_resolution(0);
                double grid_cell_y = (-grid_map.cols() / 2 + j) * grid_resolution(1);
                double grid_cell_z = grid_map(i, j);

                if (std::isnan(grid_cell_z))
                {
                    // we skip already occluded cells to improve computational efficiency
                    occlusion_mask(i, j) = true;
                    continue;
                }

                elevation_map(i, j) = grid_map(i, j);

                Eigen::Vector3d grid_cell_pos = {grid_cell_x, grid_cell_y, grid_cell_z};

                Eigen::Vector3d direction = grid_cell_pos - vantage_point;
                double ray_length = direction.norm();
                direction /= ray_length;

                Eigen::Vector3d raycast_pos = vantage_point;
                bool grid_cell_occluded = false;
                while (grid_cell_occluded == false)
                {
                    raycast_pos += 0.5 * direction * std::min(grid_resolution(0), grid_resolution(1));

                    int raycast_u = (int)std::round(grid_map.rows() / 2 + raycast_pos(0) / grid_resolution(0));
                    int raycast_v = (int)std::round(grid_map.cols() / 2 + raycast_pos(1) / grid_resolution(1));

                    // the grid_cell cannot occlude itself, thats why we do not consider the final cell
                    if ((i == raycast_u && j == raycast_v))
                    {
                        // std::cout << "break because we reached grid_cell for i=" << i << ", j=" << j << std::endl;
                        // elevation_map(raycast_u, raycast_v) = grid_map(raycast_u, raycast_v);
                        break;
                    }

                    if ((raycast_pos - vantage_point).norm() > ray_length)
                    {
                        // std::cout << "break because we are past max_distance for i=" << i << ", j=" << j << std::endl;
                        break;
                    }

                    double ground_elevation = grid_map(raycast_u, raycast_v);

                    elevation_map(raycast_u, raycast_v) = std::min(elevation_map(raycast_u, raycast_v), raycast_pos[2]);

                    // we do not consider cells with missing elevation information to cause occlusion
                    if (!std::isnan(ground_elevation))
                    {
                        // we consider a cell to be occluded, if the ray hits a higher elevation on its trajectory to the cell
                        if (ground_elevation > raycast_pos(2))
                        {
                            elevation_map(raycast_u, raycast_v) = ground_elevation;
                            grid_cell_occluded = true;
                            // std::cout << "break because we found occluded grid_cell for u=" << raycast_u << ", v=" << raycast_v << std::endl;
                        }
                    }
                }

                occlusion_mask(i, j) = grid_cell_occluded;
            }
        }

        return elevation_map;
    }

}

PYBIND11_MODULE(grid_map_raycasting, m)
{
    m.doc() = R"pbdoc(
        C++ component including Python bindings to raycast a gridmap from a viewpoint to check for occlusions
        -----------------------

        .. currentmodule:: grid_map_raycasting

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("createElevationMap", &grid_map_raycasting::createElevationMap, R"pbdoc(
        Create a elevation map from a point cloud.
    )pbdoc",
          py::arg("points"), py::arg("voxel_size"), py::arg("width"), py::arg("height"), py::arg("origin"));

    m.def("rayCastGridMap", &grid_map_raycasting::rayCastGridMap, R"pbdoc(
        Raycast every cell on the grid from a constant origin of the ray.

        It returns a grid map of booleans which signify weather the grid cell is visible from the vantage point of the robot or if its hidden by the terrain.
        Formulated alternatively, it creates an occlusion mask for a given Digital Elevation Map (DEM) which stores true for occluded and false for visible.
    )pbdoc",
          py::arg("vantage_point"), py::arg("grid_map"), py::arg("grid_resolution"));

    m.def("rayCastElevationMap", &grid_map_raycasting::rayCastElevationMap, R"pbdoc(
        Raycast every cell on the grid from a constant origin of the ray.

        It returns a grid map of double value which represents max height of each grid.
        If the value is std::numeric_limits::max(), its grid is considered as an occlusion.
    )pbdoc",
          py::arg("vantage_point"), py::arg("grid_map"), py::arg("grid_resolution"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}