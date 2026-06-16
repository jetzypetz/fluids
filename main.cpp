#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <sstream>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "lbfgs.h"

double sqr(double x) { return x * x; };

class Vector {
    public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[2];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}

double abs_cross_prod(Vector a, Vector b) {
    return std::abs(a[0] * b[1] - a[1] * b[0]);
}

class Polygon {
    public:
    
    double area() {
        if (vertices.size() < 3) return 0;
        // TODO Lab 3
        // Compute the area of the polygon
        
        double area = 0;
        
        for (size_t i=0; i<vertices.size(); i++) {
            size_t next_i = (i+1) % vertices.size();
            area += (vertices[i][0] * vertices[next_i][1])
                  - (vertices[next_i][0] * vertices[i][1]);
        }

        area = std::abs(area) * 1./2;
        
        return area;
    }
    
    Vector centroid() {
        if (vertices.size() < 3) return Vector(0, 0);
        // TODO Lab 3
        // Compute the centroid of the polygon
        Vector C(0, 0);

        for (size_t i=0; i<vertices.size(); i++) {
            size_t next_i = (i+1) % vertices.size();
            double factor = (vertices[i][0] * vertices[next_i][1])
                          - (vertices[next_i][0] * vertices[i][1]);
            C[0] += (vertices[i][0] + vertices[next_i][0]) * factor;
            C[1] += (vertices[i][1] + vertices[next_i][1]) * factor;
        }

        C = C * 1/(6 * area());

        return C;
    }

    double integral_square_distance(const Vector& Pi) {
        if (vertices.size() < 3) return 0;
        
        // TODO Lab 3
        // Compute the integral of ||x-Pi||^2 over the polygon
        
        double integral = 0.0;

        for (size_t j=1; j + 1 < vertices.size(); j++) {
            // triangle is (0, j, j+1)
            std::vector<size_t> c = {0, j, j+1};
            double abs_T = abs_cross_prod(vertices[c[1]] - vertices[c[0]],
                                          vertices[c[2]] - vertices[c[0]]) / 2;
            double triangle_integral = 0;
            for (size_t k=0; k<3; k++) {
                for (size_t l=k; l<3; l++) {
                    triangle_integral +=
                            dot(vertices[c[k]] - Pi,
                                vertices[c[l]] - Pi);
                }
            }
            integral += abs_T * triangle_integral / 6;
        }

        return integral;
    }
    
    std::vector<Vector> vertices;
};

void save_frame(std::vector<Polygon>& cells, std::string filename, int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3) continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
        for (int j = 0; j < n; ++j) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;

                int prev_sign = 0;
                bool isInside = true;
                bool isEdge = false;

                for (int j = 0; j < n; ++j) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];
                    double dx = bx - ax, dy = by - ay;
                    double qx = px - ax, qy = py - ay;
                    
                    double det = qx * dy - qy * dx;
                    int s = (det > 1e-12) - (det < -1e-12);

                    if (s != 0) {
                        if (prev_sign != 0 && s != prev_sign) {
                            isInside = false;
                            break;
                        }
                        prev_sign = s;
                    }

                    double len2 = dx * dx + dy * dy;
                    double dot = qx * dx + qy * dy;
                    if (dot >= 0.0 && dot <= len2 && det * det <= edge_width2 * len2)
                    isEdge = true;
                }
                
                if (isInside) {
                    int id = (H - 1 - y) * W + x;
                    inside[id] = 1;
                    if (isEdge) edge[id] = 1;
                }
            }
        }
    }
    
    std::vector<unsigned char> image(W * H * 3, 255);

    #pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        if (edge[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        }
        else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    // Draw centroids as red circles (by ai)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        if (V.size() < 3) continue;
        
        // Get centroid coordinates
        Vector centroid = cells[i].centroid();
        double cx = centroid[0] * W;
        double cy = centroid[1] * H;
        
        // Convert to pixel coordinates (with y-flip)
        int center_x = (int)std::round(cx);
        int center_y = H - 1 - (int)std::round(cy);
        
        // Circle radius
        int radius = 4;
        int radius_sq = radius * radius;
        
        // Draw circle using Bresenham-inspired approach
        int x0 = std::max(0, center_x - radius);
        int x1 = std::min(W - 1, center_x + radius);
        int y0 = std::max(0, center_y - radius);
        int y1 = std::min(H - 1, center_y + radius);
        
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                int dx = x - center_x;
                int dy = y - center_y;
                if (dx*dx + dy*dy <= radius_sq) {
                    int idx = y * W + x;
                    // Override any existing color (edges or cell interiors)
                    image[3 * idx + 0] = 0;
                    image[3 * idx + 1] = 255;
                    image[3 * idx + 2] = 0;
                }
            }
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}

Vector intersect(const Vector& prevVertex, const Vector& curVertex, const Vector& P0, const Vector& Pi, double w0, double wi) {
    double denominator = dot(curVertex - prevVertex, Pi - P0);
    double t;

    // probably a better way
    if (denominator == 0) {
        t = __DBL_MAX__;
        return prevVertex;
    }

    Vector M = (P0 + Pi) / 2;
    Vector M_prime = M + ((w0 - wi) / (2 * dot(P0 - Pi, P0 - Pi))) * (Pi - P0);

    t = dot(M_prime - prevVertex, Pi - P0) / denominator;

    return prevVertex + t * (curVertex - prevVertex);
}

Vector intersect_edge(const Vector& prevVertex, const Vector& curVertex, const Vector& u, const Vector& v, Vector& N) {
    N = Vector(v[1] - u[1], u[0] - v[0]);
    double denominator = dot(curVertex - prevVertex, N);
    double t;

    // probably a better way
    if (denominator == 0) {
        t = __DBL_MAX__;
        return prevVertex;
    }

    t = dot(u - prevVertex, N) / denominator;
    return prevVertex + t * (curVertex - prevVertex);
}

bool is_inside_edge(const Vector& p, const Vector& u, const Vector& v, const Vector& N) {
    return dot(u - p, N) <= 0;
}

class VoronoiDiagram {

    public:
    
    VoronoiDiagram() {
    };

    void compute() {

        cells.resize(points.size());

        Polygon square;
        square.vertices.push_back(Vector(0, 0));
        square.vertices.push_back(Vector(1, 0));
        square.vertices.push_back(Vector(1, 1));
        square.vertices.push_back(Vector(0, 1));

        // TODO Lab 1 (Voronoi)
        // For all sites Pi (in parallel) :
        //      Start with a unit square
        //      For all other sites Pj (optionally, only k nearest neighbors) :
        //          Clip it with bisector of [Pi,Pj]
        //      (Lab 3, fluids) : also clip it by a disk of radius sqrt(w_i - w_air) centered at Pi
        for (size_t i=0; i<points.size(); i++) {
            
            Polygon cell = square;
            
            for (size_t j=0; j<points.size(); j++) {
                if (j==i) { continue; }
                cell = clip_by_bisector(cell, points[i], points[j], weights[i], weights[j]);
            }
            
            double r = std::sqrt(weights[i] - weights[points.size()]);
            
            size_t M = 50;
            Vector centroid = cell.centroid();
            
            double theta0 = 0;
            double theta1 = 0;

            for (size_t j=1; j<M+1; j++) {
                theta0 = theta1;
                theta1 = 2.0 * M_PI * j / M;

                // Vector u = points[i] + r * Vector(std::sin(theta0), std::cos(theta0));
                // Vector v = points[i] + r * Vector(std::sin(theta1), std::cos(theta1));

                Vector u = centroid + r * Vector(std::sin(theta0), std::cos(theta0));
                Vector v = centroid + r * Vector(std::sin(theta1), std::cos(theta1));

                cell = clip_by_edge(cell, u, v);
            }
            cells[i] = cell;
        }
    
    }

    static Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {

        // TODO Lab 3 (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a (discretized) disk
        
        Polygon result;

        for (size_t i=0; i<V.vertices.size(); i++) {
            Vector curVertex = V.vertices[i];
            Vector prevVertex = V.vertices[(i>0) ? (i-1) : (V.vertices.size() - 1)];
            
            Vector N;
            Vector intersection = intersect_edge(prevVertex, curVertex, u, v, N);
            
            if (is_inside_edge(curVertex, u, v, N)) {
                if (!is_inside_edge(prevVertex, u, v, N)) {
                    result.vertices.push_back(intersection);
                }
                result.vertices.push_back(curVertex);
            } else {
                if (is_inside_edge(prevVertex, u, v, N)) {
                    result.vertices.push_back(intersection);
                }
            }

        }
        return result;
    }

    static Polygon clip_by_bisector(const Polygon& V, const Vector& P0, const Vector& Pi, double w0, double wi) {
        
        // TODO Lab 1 (Voronoi) : in Lab 1, we assume w0 = w1 = 0
        // Clip a polygon by the bisector of the segment defined by P0 (the current site of the Voronoi cell being computed) and Pi (another site)
        
        Polygon result;
        
        for (size_t i=0; i<V.vertices.size(); i++) {
            Vector curVertex = V.vertices[i];
            Vector prevVertex = V.vertices[(i>0) ? (i-1) : (V.vertices.size() - 1)];
            
            Vector intersection = intersect(prevVertex, curVertex, P0, Pi, w0, wi);
            
            if (dot(curVertex - P0, curVertex - P0) - dot(curVertex - Pi, curVertex - Pi) - w0 + wi <= 0) {
                if (dot(prevVertex - P0, prevVertex - P0) - dot(prevVertex - Pi, prevVertex - Pi) - w0 + wi > 0) {
                    result.vertices.push_back(intersection);
                }
                result.vertices.push_back(curVertex);
            } else {
                if (dot(prevVertex - P0, prevVertex - P0) - dot(prevVertex - Pi, prevVertex - Pi) - w0 + wi <= 0) {
                    result.vertices.push_back(intersection);
                }
            }

        }
        
        // TODO Lab 2 (Semi-Discrete Optimal Transport) : extend to Laguerre cells, i.e., w0 != w1
        
        return result;
    }

    double desired_fluid;
    
    std::vector<Vector> points;    // Lab 1 (Voronoi) : the sites to consider
    
    std::vector<double> weights;   // Lab 2 (OT) : the weight associated to each site (the Laguerre weight, i.e. the dual optimal transport variables to be optimized)
    
    std::vector<Polygon> cells;   // Lab 1 : the polygons representing each individual cell
    
};


// Lab 2 
class OptimalTransport {
    
    public:
    OptimalTransport() {};

    void optimize();
    
    VoronoiDiagram vor;
};


// Labs 2 and 3
static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
)
{
    OptimalTransport* ot = (OptimalTransport*)(instance);
    
    // first compute the Voronoi diagram at the current optimization step
    memcpy(&ot->vor.weights[0], x, n * sizeof(x[0]));
    ot->vor.compute();
  
    
    // Lab 2 (Optimal transport) : compute the function to be minimized (fx) and its gradient (g[i], i=0..n-1)
    // Lab 3 (fluid) : adapt these functions to support partial optimal transport (now "n" has been increased by 1 to account for the air variable)
    
    lbfgsfloatval_t fx = 0.0;
    size_t N = n - 1;

    double desired_fluid = ot->vor.desired_fluid;
    double estimated_fluid = 0;
    for (size_t i=0; i<N; i++) {
        Polygon& polygon = ot->vor.cells[i];
        Vector  point    = ot->vor.points[i];
        
        double area = polygon.area();
        double isd  = polygon.integral_square_distance(point);
        
        estimated_fluid += area;
        
        fx  += (x[i] * (area - desired_fluid/N)) - isd;
        g[i] = area - (desired_fluid/N);
    }

    double desired_air   = 1 - desired_fluid;
    double estimated_air = 1 - estimated_fluid;

    fx  += x[N] * (estimated_air - desired_air);
    g[N] = estimated_air - desired_air;

    return fx;
}

// Labs 2 and 3 : you may use this function to print debugging info.
static int progress(
    void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
    int n, int k, int ls) {
    printf("Iteration %d:\n", k);
    printf("  fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}


// Lab 2
void OptimalTransport::optimize() {
    
    lbfgsfloatval_t fx;
    std::vector<double> weights(vor.weights);
    
    lbfgs_parameter_t param;
    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&param);

    // run the LBFGS optimizer

    if (int ret = lbfgs(weights.size(), &weights[0], &fx, evaluate, progress, (void*)this, &param)) {
        std::cout << "failed lbfgs, return code: " << ret << std::endl;
    } else {
        std::cout << "passed lbfgs" << std::endl;
    }

    // copy the result back to the voronoi structure
    vor.weights = weights;
    
    // finally recompute the Voronoi diagram with the final optimized weights
    vor.compute();
}


// Lab 3 (fluids)
class Fluid {
    public:
    Fluid(size_t N_particles = 1000) : N_particles(N_particles) {
        
        Vector c(0.5, 0.6);
        double r = std::sqrt(desired_fluid / M_PI);
        
        while (particles.size() < N_particles) {
            double x = rand() / (double) RAND_MAX;
            double y = rand() / (double) RAND_MAX;
            
            Vector p(x, y);
            Vector d = p - c;
            
            if (d.norm2() <= r * r) {
                particles.push_back(Vector(x, y));
                velocities.push_back(Vector(0,0));
            }
        }
        
        ot.vor.desired_fluid = desired_fluid;
        ot.vor.points = particles;
        ot.vor.weights.assign(N_particles, 0.01);
        ot.vor.weights.push_back(0.0);
    }
    
    // Lab 3 : advance the simulation dt in time
    void time_step(double dt) {
        
        double epsilon2 = 0.004 * 0.004;
        Vector g(0, -9.81);
        double m_i = 200;
        
        // TODO Lab 3 : 
        // Compute semi-discrete partial optimal transport
        // for all particles, add gravity and spring force towards cell centroid, integrate acceleration->velocity and velocity->position
        ot.optimize();
        particles = ot.vor.points;

        for (size_t i=0; i<N_particles; i++) {
            Vector F_spring = (ot.vor.cells[i].centroid() - particles[i]) / epsilon2;
            Vector F        = F_spring + (m_i * g);

            velocities[i] = velocities[i] + (dt * F / m_i);
            particles[i] = particles[i] + (dt * velocities[i]);

            if (particles[i][0] < 0.0) {
                particles[i][0] = 0.0;
                if (velocities[i][0] < 0.0) velocities[i][0] *= -0.5;
            }
            if (particles[i][0] > 1.0) {
                particles[i][0] = 1.0;
                if (velocities[i][0] > 0.0) velocities[i][0] *= -0.5;
            }
            if (particles[i][1] < 0.0) {
                particles[i][1] = 0.0;
                if (velocities[i][1] < 0.0) velocities[i][1] *= -0.5;
            }
            if (particles[i][1] > 1.0) {
                particles[i][1] = 1.0;
                if (velocities[i][1] > 0.0) velocities[i][1] *= -0.5;
            }
        }
        
        ot.vor.points = particles;
    }
    
    // just run the full simulation
    void run_simulation() {
        double dt = 0.002;
        for (int i = 0; i < 1000; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "test", i);
        }
    }

    size_t N_particles;
    OptimalTransport ot;
    std::vector<Vector> particles;  // the position of all particles
    std::vector<Vector> velocities; // the velocities of all particles
    double desired_fluid = 0.4; // you decide the fraction of the unit square occupied by the fluid
};

// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon>& polygons, std::string filename, const std::vector<Vector>* points = NULL, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (size_t i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (size_t j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }

    if (points) {
        fprintf(f, "<g>\n");
        for (size_t i = 0; i < points->size(); i++) {
            fprintf(f, "<circle cx = \"%3.3f\" cy = \"%3.3f\" r = \"3\" />\n", (*points)[i][0] * 1000., 1000. - (*points)[i][1] * 1000);
        }
        fprintf(f, "</g>\n");
        
    }
    
    fprintf(f, "</svg>\n");
    fclose(f);
}


int main() {

    Fluid fluid(200);
    fluid.run_simulation();

    return 0;
}