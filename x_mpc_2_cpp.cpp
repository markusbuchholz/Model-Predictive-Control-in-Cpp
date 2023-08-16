// Markus Buchholz, 2023
// g++ x_mpc_1_cpp.cpp -o t -I/usr/include/python3.8 -lpython3.8

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXd;

// MPC parameters
int horizon = 10; // Prediction horizon
float Q = 3.0f;   // State cost
float R = 1.5f;   // Control cost

//------------------------------------------------------------------------------------

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> run_mpc()
{
    // Simulation parameters
    std::vector<float> curve;
    int mpc_loop = 50;
    float initial_state = 0.1f;
    double desired_state = 25.0f;

    std::vector<float> time;
    // Model parameters
    VectorXd A(1); // state transition matrix
    VectorXd B(1); // control matrix

    A << 0.9;
    B << 0.5;
    std::vector<float> states;

    double state = 0.0f;

    std::vector<float> controls;
    double control_input = 0.0f;
    state = initial_state;
    // time.push_back(0.0);

    for (int ii = 0; ii < mpc_loop; ii++)
    {
        time.push_back(ii);
        // float cxi = ii * ii * 0.05; //ok1
        float cxi = std::sin((float)ii * M_PI/18.0);

        desired_state = cxi;

        curve.push_back(cxi);
        VectorXd predicted_states = VectorXd::Zero(horizon + 1);
        predicted_states[0] = state;

        // states.push_back((float)state);

        for (int jj = 1; jj < horizon; jj++)
        {

            predicted_states[jj] = A[0] * predicted_states[jj - 1] + B[0] * control_input;
        }

        VectorXd control_errors = VectorXd::Zero(horizon);
        VectorXd state_errors = VectorXd::Zero(horizon + 1);

        for (int ii = 0; ii < predicted_states.size(); ii++)
        {

            state_errors[ii] = predicted_states[ii] - desired_state;
        }

        VectorXd f = VectorXd::Zero(state_errors.size() + control_errors.size());

        f << state_errors, control_errors;

        MatrixXd Q_matrix = Eigen::MatrixXd::Zero(horizon + 1, horizon + 1);
        Q_matrix.diagonal() = Eigen::VectorXd::Constant(horizon + 1, Q);

        MatrixXd R_matrix = Eigen::MatrixXd::Zero(horizon, horizon);
        R_matrix.diagonal() = Eigen::VectorXd::Constant(horizon, R);

        MatrixXd H(21, 21);
        H.setZero();

        H.block(0, 0, Q_matrix.rows(), Q_matrix.cols()) = Q_matrix;
        H.block(11, 11, R_matrix.rows(), R_matrix.cols()) = R_matrix;

        // Solve H * control_input_delta = concatenaed
        Eigen::VectorXd control_input_delta = H.colPivHouseholderQr().solve(-f);

        control_input += control_input_delta[0]; // only first affect the decision

        state = A[0] * state + B[0] * control_input;

        states.push_back((float)state);
        controls.push_back((float)control_input);
    }

    std::cout << time.size() << ": " << states.size() << " :" << controls.size() << "\n";
    return std::make_tuple(time, states, controls, curve);
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