#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include <assert.h>

#define _USE_MATH_DEFINES
#include <math.h>

#define DEBUG 1

enum DimName_t
{
    DimX,
    DimY,
    DimZ
};

static const int ndim = 3;
static const int time_layers = 3;

static int N, M;
static double hx, hy, hz;
static double tau = 0.001;
static double tau_2 = tau * tau;
static double Lx, Ly, Lz;
static int dx, dy, dz;
static int rank = -1;

static int index( int i, int j, int k) { return i + j * dx + k * dx * dy; }

static double
laplaceOperator( const double data[], int i, int j, int k)
{
    double val = 0;
    int p_prev, p_curr = index(i, j, k), p_next;

    p_prev = index(i-1, j, k);
    p_next = index(i+1, j, k);
    val += (data[p_prev] - 2 * data[p_curr] + data[p_next]) / hx / hx;

    p_prev = index(i, j-1, k);
    p_next = index(i, j+1, k);
    val += (data[p_prev] - 2 * data[p_curr] + data[p_next]) / hy / hy;

    p_prev = index(i, j, k-1);
    p_next = index(i, j, k+1);
    val += (data[p_prev] - 2 * data[p_curr] + data[p_next]) / hz / hz;

    return val;
} /* laplaceOperator */


static double x( int i) { return hx * i; }
static double y( int j) { return hy * j; }
static double z( int k) { return hz * k; }

static double
phi_func( double x, double y, double z)
{
    return sin((M_PI * x) / Lx) * sin((M_PI * y) / Ly) * sin((M_PI * z) / Lz);
} /* phi_func */

static double
u_func( double x, double y, double z, double t)
{
    static double a = M_PI * std::sqrt(1.0/(Lx*Lx) + 1.0/(Ly*Ly) + 1.0/(Lz*Lz));
    return cos(a * t) * phi_func(x, y, z);
} /* u_func */

static void
setBoarders( double *data, DimName_t dname, bool is_first, bool is_last)
{
    if ( !is_first && !is_last )
    {
        return;
    }

    int dim1 = dname == DimX ? dy : dx; 
    int dim2 = dname == DimZ ? dy : dz;
    for ( int p = 0; p < dim1 * dim2; ++p )
    {
        int it1 = p % dim1;
        int it2 = p / dim1;

        if ( is_first )
        {
            switch ( dname )
            {
              case DimX:
                data[index(1, it1, it2)] = 0;
                break;
              case DimY:
                data[index( it1, 1, it2)] = 0;
                break;
              case DimZ:
                data[index( it1, it2, 1)] = 0;
                break;                
            }
        }

        if ( is_last )
        {
            switch ( dname )
            {
              case DimX:
                data[index( dx - 2, it1, it2)] = 0;
                break;
              case DimY:
                data[index( it1, dy - 2, it2)] = 0;
                break;
              case DimZ:
                data[index( it1, it2, dz - 2)] = 0;
                break;                
            }
        }
    }
} /* setBoarders */

static void
sendForward( double *data, DimName_t dname, MPI_Comm& comm_cart,
             int rank_prev, int rank_next, bool is_first, bool is_last)
{
    const int dim1 = dname == DimX ? dy : dx;
    const int dim2 = dname == DimZ ? dy : dz;
    const int size = dim1 * dim2;

    if ( is_first && is_last )
    {
        for ( int p = 0; p < size; p++ )
        {
            int it1 = p % dim1;
            int it2 = p / dim1;

            switch ( dname )
            {
              case DimX:
                data[index(0, it1, it2)] = data[index(dx - 2, it1, it2)];
                break;
              case DimY:  
                data[index(it1, 0, it2)] = data[index(it1, dy - 2, it2)];
                break;
              case DimZ:
                data[index(it1, it2, 0)] = data[index(it1, it2, dz - 2)];
                break;
            }
        }
        return;
    }

    MPI_Status comm_status;
    double send_buffer[size], recv_buffer[size];

    for ( int p = 0; p < size; p++ )
    {
        int it1 = p % dim1;
        int it2 = p / dim1;

        switch ( dname )
        {
          case DimX:  
            send_buffer[p] = data[index(dx - 2, it1, it2)];
            break;
          case DimY:  
            send_buffer[p] = data[index(it1, dy - 2, it2)];
            break;
          case DimZ:  
            send_buffer[p] = data[index(it1, it2, dz - 2)];
            break;
        }
    }

    int tag = dname + 1;
    MPI_Sendrecv( send_buffer, size, MPI_DOUBLE, rank_next, tag,
                  recv_buffer, size, MPI_DOUBLE, rank_prev, tag,
                  comm_cart, &comm_status);

    for ( int p = 0; p < size; p++ )
    {
        int it1 = p % dim1;
        int it2 = p / dim1;

        switch ( dname )
        {
          case DimX:  
            data[index(0, it1, it2)] = recv_buffer[p];
            break;
          case DimY:  
            data[index(it1, 0, it2)] = recv_buffer[p];
            break;
          case DimZ:  
            data[index(it1, it2, 0)] = recv_buffer[p];
            break;
        }
    }
} /* sendForward */

static void
sendBackward( double *data, DimName_t dname, MPI_Comm& comm_cart,
              int rank_prev, int rank_next, bool is_first, bool is_last)
{
    const int dim1 = dname == DimX ? dy : dx;
    const int dim2 = dname == DimZ ? dy : dz;
    const int size = dim1 * dim2;

    if ( is_first && is_last )
    {
        return;
    }

    MPI_Status comm_status;

    double send_buffer[size], recv_buffer[size];

    for ( int p = 0; p < size; p++ )
    {
        int it1 = p % dim1;
        int it2 = p / dim1;

        switch ( dname )
        {
          case DimX:
            send_buffer[p] = data[index( 1, it1, it2)];
            break;
          case DimY:  
            send_buffer[p] = data[index( it1, 1, it2)];
            break;
          case DimZ:  
            send_buffer[p] = data[index( it1, it2, 1)];
            break;
        }
    }

    int tag = dname + 4;
    MPI_Sendrecv( send_buffer, size, MPI_DOUBLE, rank_prev, tag,
                  recv_buffer, size, MPI_DOUBLE, rank_next, tag,
                  comm_cart, &comm_status);


    for ( int p = 0; p < size; p++ )
    {
        int it1 = p % dim1;
        int it2 = p / dim1;

        switch ( dname )
        {
          case DimX:  
            data[index(dx - 1, it1, it2)] = recv_buffer[p];
            break;
          case DimY:  
            data[index(it1, dy - 1, it2)] = recv_buffer[p];
            break;
          case DimZ:  
            data[index(it1, it2, dz - 1)] = recv_buffer[p];
            break;
        }
    }
} /* sendBackward */

struct EstimateError
{
    double mse;
    double max;

    EstimateError() : mse(0), max(0) {}
};

static void 
estimateError( EstimateError* p_error, const double *data, int step,
               int i_min, int j_min, int k_min)
{
    double mse = 0;
    double max = 0;

#pragma omp parallel for reduction(+:mse)
    for (int p = 0; p < dx * dy * dz; p++)
    {
        int i = p % dx;
        int j = (p / dx) % dy;
        int k = (p / dx / dy) % dz;

        // Skip halo
        if ( i == 0 || i == dx - 1
             || j == 0 || j == dy - 1
             || k == 0 || k == dz - 1 )
        {
            continue;
        }

        double u_true = u_func(x(i_min + i - 1), y(j_min + j - 1), z(k_min + k - 1), step * tau);
        double u_pred = data[p];
        mse += pow(u_true - u_pred, 2);

#pragma omp critical
        {
            double u_abs = fabs(u_true - u_pred);
            max = u_abs > max ? u_abs : max;
        }
    }

    p_error->max = max;
    p_error->mse = mse;
}

/**
 * Execute:
 * $ ./a.out [N] [M] [Lx] [Ly] [Lz]
 */
int main( int argc, char **argv)
{
    MPI_Init( &argc, &argv);

    int nproc = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &nproc);

    N =  argc > 1 ? std::atoi(argv[1]) : 128;
    M =  argc > 2 ? std::atoi(argv[2]) : time_layers;
    Lx = argc > 3 ? std::atof(argv[3]) : 1;
    Ly = argc > 4 ? std::atof(argv[4]) : Lx;
    Lz = argc > 5 ? std::atof(argv[5]) : Ly;

    hx = Lx / N; 
    hy = Ly / N;
    hz = Lz / N;

    int dims[ndim] = {};
    MPI_Dims_create( nproc, ndim, dims);

    int nodes[ndim];
    for ( int d = 0; d < ndim; ++d )
    {
        nodes[d] = ceil((double)(N + 1) / dims[d]);
        if ( nodes[d] == 0 ) {
            std::cerr << "[ERROR] Invalid grid split" << std::endl;
            return 1;
        }
    }

    double gTime = MPI_Wtime();
    
    // Grid info
    if (
         DEBUG && 
         rank == 0 )
    {
        std::cout << N << ' ' << M << ' ' << nproc << std::endl;
        for (int d = 0; d < ndim; d++) {
            std::cout << "axis" << d << '\t'
                      << dims[d] << '\t' << nodes[d] << std::endl;
        }
    }

    MPI_Comm comm_cart;
    int periods[ndim] = {};
    MPI_Cart_create( MPI_COMM_WORLD, ndim, dims, periods, 0, &comm_cart);

    int coords[ndim];
    MPI_Cart_coords( comm_cart, rank, ndim, coords);

    int rank_prev[ndim], rank_next[ndim];
    for ( int d = 0; d < ndim; d++ )
    {
        MPI_Cart_shift( comm_cart, d, 1, &rank_prev[d], &rank_next[d]);
    }

    bool is_first[ndim], is_last[ndim];
    for ( int d = 0; d < ndim; d++ )
    {
        is_first[d] = (coords[d] == 0);
        is_last[d] =  (coords[d] == dims[d] - 1);
    }

    const int i_min = coords[0] * nodes[0];
    const int i_max = std::min(N+1, (coords[0] + 1) * nodes[0]) - 1;

    const int j_min = coords[1] * nodes[1];
    const int j_max = std::min(N+1, (coords[1] + 1) * nodes[1]) - 1;

    const int k_min = coords[2] * nodes[2];
    const int k_max = std::min(N+1, (coords[2] + 1) * nodes[2]) - 1;

    dx = i_max - i_min + 1 + 2; // + 2 for halo
    dy = j_max - j_min + 1 + 2;
    dz = k_max - k_min + 1 + 2;

    EstimateError error_total, error_cur, error_proc;

    double* d_array[time_layers];
    for (int p = 0; p < time_layers; p++)
    {
        d_array[p] = new double[dx * dy * dz];
    }

    // Fill T = 0
#pragma omp parallel for
    for (int p = 0; p < dx * dy * dz; p++)
    {
        int i = p % dx;
        int j = (p / dx) % dy;
        int k = p / dx / dy;
        d_array[0][p] = phi_func(x(i_min + i - 1), y(j_min + j - 1), z(k_min + k - 1));
    }

    if (DEBUG)
    {
        error_cur.mse = 0;
        error_cur.max = 0;

        estimateError(&error_proc, d_array[0], 0, i_min, j_min, k_min);

        MPI_Reduce(&error_proc.mse, &error_cur.mse, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
        MPI_Reduce(&error_proc.max, &error_cur.max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);

        if ( rank == 0 )
        {
            error_cur.mse /= pow(N+1, 3);
            error_total.mse += error_cur.mse;

            error_total.max = error_cur.max > error_total.max ? error_cur.max : error_total.mse;
        }
    }

    if ( rank == 0 )
    {
        printf("[iter %03d]", 0);
        if (DEBUG)
        {
            printf(" RMSE = %.6f; MAX = %.6f;", sqrt(error_cur.mse), error_cur.max);
        }
        printf(" Time = %.6f sec.\n", MPI_Wtime() - gTime);
    }

    // General case
    for ( int T = 1; T < M; T++ )
    {
        double iter_time = MPI_Wtime();
#pragma omp parallel for
        for ( int p = 0; p < dx * dy * dz; p++ )
        {
            int i = p % dx;
            int j = (p / dx) % dy;
            int k = (p / dx / dy) % dz;

            // Skip halo
            if ( i == 0 || i == dx - 1
                 || j == 0 || j == dy - 1
                 || k == 0 || k == dz - 1 )
            {
                continue;
            }

            // Skip boarders
            if ( is_first[0] && i == 1 || is_last[0] && i == dx - 2
                 || is_first[1] && j == 1 || is_last[1] && j == dy - 2
                 || is_first[2] && k == 1 || is_last[2] && k == dz - 2 )
            {
                continue;
            }

            // Special case for T == 1
            if ( T == 1 )
            {
                d_array[T][p] = d_array[T-1][p] + 0.5 * tau_2 * (laplaceOperator(d_array[T-1], i, j, k));
            } else
            {
                d_array[T % time_layers][p] = 2 * d_array[(T - 1) % time_layers][p] - d_array[(T - 2) % time_layers][p]
                                              + tau_2 * ( laplaceOperator(d_array[(T - 1) % time_layers], i, j, k));
            }
        }

        // Send interfaces
        sendForward (d_array[T % time_layers], DimX, comm_cart, rank_prev[0], rank_next[0], is_first[0], is_last[0]);
        sendBackward(d_array[T % time_layers], DimX, comm_cart, rank_prev[0], rank_next[0], is_first[0], is_last[0]);

        sendForward (d_array[T % time_layers], DimY, comm_cart, rank_prev[DimY], rank_next[DimY], is_first[DimY], is_last[DimY]);
        sendBackward(d_array[T % time_layers], DimY, comm_cart, rank_prev[DimY], rank_next[DimY], is_first[DimY], is_last[DimY]);

        sendForward (d_array[T % time_layers], DimZ, comm_cart, rank_prev[DimZ], rank_next[DimZ], is_first[DimZ], is_last[DimZ]);
        sendBackward(d_array[T % time_layers], DimZ, comm_cart, rank_prev[DimZ], rank_next[DimZ], is_first[DimZ], is_last[DimZ]);

        // Set boarders
        setBoarders(d_array[T % time_layers], DimX, is_first[DimX], is_last[DimX]);
        setBoarders(d_array[T % time_layers], DimY, is_first[DimY], is_last[DimY]);
        setBoarders(d_array[T % time_layers], DimZ, is_first[DimZ], is_last[DimZ]);

        if ( DEBUG )
        {
            error_cur.mse = 0;
            error_cur.max = 0;

            estimateError(&error_proc, d_array[T % time_layers], T, i_min, j_min, k_min);

            MPI_Reduce(&error_proc.mse, &error_cur.mse, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
            MPI_Reduce(&error_proc.max, &error_cur.max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);

            if ( rank == 0 )
            {
                error_cur.mse /= pow(N, 3);
                error_total.mse += error_cur.mse;

                error_total.max = error_cur.max > error_total.max ? error_cur.max : error_total.max;
            }
        }

        if ( rank == 0 )
        {
            printf("[iter %03d]", T);
            if ( DEBUG )
            {
                printf(" RMSE = %.6f; MAX = %.6f;", sqrt(error_cur.mse), error_cur.max);
            }
            printf(" Time = %.6f sec.\n", MPI_Wtime() - iter_time);
        }
    }

    if ( rank == 0
         && DEBUG )
    {
        printf("Final RMSE = %.6f; MAX = %.6f\n", sqrt(error_total.mse / M), error_total.max);
        printf("Task elapsed in: %.6f sec.\n", MPI_Wtime() - gTime);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for ( int l = 0; l < time_layers; l++ )
    {
        delete d_array[l];
    }

    if ( rank == 0 )
    {
        printf("\n");
        printf("%d processes %d knots %.6f Lz\n", nproc, N, Lz);
        printf("Time total:     %.6f\n", MPI_Wtime() - gTime);
    }

    MPI_Finalize();
    return 0;
}
