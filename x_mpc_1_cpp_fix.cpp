// Markus Buchholz, 2024
// g++ x_mpc_1_cpp.cpp -o t -I/usr/include/eigen3 -I/usr/include/python3.12 -I/usr/lib/python3/dist-packages/numpy/core/include -lpython3.12

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using Eigen::MatrixXd;
using Eigen::VectorXd;

// MPC parameters
int horizon = 10; // Prediction horizon
float Q = 1.0f;   // State cost
float R = 0.1f;   // Control cost

//------------------------------------------------------------------------------------

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> run_mpc()
{
    // Simulation parameters
    int mpc_loop = 50;
    float initial_state = 20.0f;
    double desired_state = 25.0f;

    std::vector<float> time;
    // Model parameters
    double A = 0.9; // state transition scalar
    double B = 0.5; // control scalar

    std::vector<float> states;
    std::vector<float> controls;

    double state = initial_state;
    double control_input = 0.0f;

    for (int ii = 0; ii < mpc_loop; ii++)
    {
        time.push_back(ii);

        // State prediction
        VectorXd predicted_states = VectorXd::Zero(horizon + 1);
        predicted_states[0] = state;

        for (int jj = 1; jj <= horizon; jj++)
        {
            predicted_states[jj] = A * predicted_states[jj - 1] + B * control_input;
        }

        // Calculate state errors
        VectorXd state_errors = VectorXd::Zero(horizon);
        for (int jj = 0; jj < horizon; jj++)
        {
            state_errors[jj] = predicted_states[jj + 1] - desired_state;
        }

        // Form the QP matrices
        MatrixXd Q_matrix = MatrixXd::Identity(horizon, horizon) * Q;
        MatrixXd R_matrix = MatrixXd::Identity(horizon, horizon) * R;

        MatrixXd H = MatrixXd::Zero(horizon, horizon);
        H.block(0, 0, horizon, horizon) = Q_matrix + R_matrix;

        VectorXd g = state_errors;

        // Solve H * control_input_delta = -g
        VectorXd control_input_delta = H.ldlt().solve(-g);

        control_input += control_input_delta[0]; // only first affect the decision

        state = A * state + B * control_input;

        states.push_back((float)state);
        controls.push_back((float)control_input);
    }

    std::cout << "Time steps: " << time.size() << ", States: " << states.size() << ", Controls: " << controls.size() << "\n";
    return std::make_tuple(time, states, controls, std::vector<float>(time.size(), desired_state));
}

//------------------------------------------------------------------------------------

void plot(std::vector<float> time, std::vector<float> states, std::vector<float> controls, std::vector<float> desired)
{
    plt::title("Model Predictive Control (MPC)");
    plt::named_plot("controls", time, controls);
    plt::named_plot("states", time, states);
    plt::named_plot("desired", time, desired, "r--");
    plt::xlabel("time");
    plt::ylabel("Y");
    plt::legend();
    plt::show();
}

//------------------------------------------------------------------------------------

int main()
{
    auto mpc = run_mpc();
    plot(std::get<0>(mpc), std::get<1>(mpc), std::get<2>(mpc), std::get<3>(mpc));
}
