/*
 * Pybind11 bindings for DEMCEM
 * 
 * Copyright (C) 2023 Niklas Wingern
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

#include "demcem_ss_ea_nxrwg.h"
#include "demcem_ss_ea_rwg.h"
#include "demcem_ss_va_nxrwg.h"
#include "demcem_ss_va_rwg.h"
#include "demcem_ws_ea_rwg.h"
#include "demcem_ws_st_rwg.h"
#include "demcem_ws_va_rwg.h"

namespace py = pybind11;

// **************************************
//     DEMCEM function declarations
// **************************************

void demcem_ss_ea_nxrwg(const double r1[], const double r2[], const double r3[], const double r4[],
                        const complex<double> ko, const int N_theta, const int N_psi,
                        complex<double> I_DE[]);
void demcem_ss_ea_rwg(const double r1[], const double r2[], const double r3[], const double r4[],
                      const complex<double> ko, const int N_theta, const int N_psi,
                      complex<double> I_DE[]);
void demcem_ss_va_nxrwg(const double r1[], const double r2[], const double r3[], const double r4[],
                        const double r5[], const complex<double> ko, const int N_theta_p,
                        const int N_theta_q, const int N_psi, complex<double> I_DE[]);
void demcem_ss_va_rwg(const double r1[], const double r2[], const double r3[], const double r4[],
                      const double r5[], const complex<double> ko, const int N_theta_p,
                      const int N_theta_q, const int N_psi, complex<double> I_DE[]);
void demcem_ws_ea_rwg(const double r1[], const double r2[], const double r3[], const double r4[],
                      const complex<double> ko, const int N_theta, const int N_psi,
                      complex<double> I_DE[]);
void demcem_ws_st_rwg(const double r1[], const double r2[], const double r3[],
                      const complex<double> ko, const int Np_1D, complex<double> I_DE[]);
void demcem_ws_va_rwg(const double r1[], const double r2[], const double r3[], const double r4[],
                      const double r5[], complex<double> ko, const int N_theta_p,
                      const int N_theta_q, const int N_psi, complex<double> I_DE[]);

// Helper function for validation checking of vertex
void check_vert(string buf_format, int buf_ndim, int buf_len)
{
    if (buf_format != py::format_descriptor<double>::format()){
        throw std::runtime_error("Vertex must be of dtype float64");
    }
    if (buf_ndim != 1){
        throw std::runtime_error("Vertex must be a 1D numpy array");
    }
    if (buf_len != 3){
        throw std::runtime_error("Vertex must have shape (3,)");
    }
}


// **************************************
//         Wrapper definitions
// **************************************

void ss_ea_nxrwg_wrapper(py::array_t<double> r1, py::array_t<double> r2, py::array_t<double> r3,
                         py::array_t<double> r4, const complex<double> ko, const int N_theta,
                         const int N_psi, py::array_t<complex<double>> result)
{
    py::buffer_info result_buf = result.request();
    if (result_buf.format != py::format_descriptor<complex<double>>::format()){
        throw std::runtime_error("Result must be of dtype complex128");
    }
    if (result_buf.ndim != 1){
        throw std::runtime_error("Result must be a 1D numpy array");
    }
    if (result_buf.shape[0] != 9){
        throw std::runtime_error("Result must have shape (9,)");
    }

    py::buffer_info r1_buf = r1.request();
    py::buffer_info r2_buf = r2.request();
    py::buffer_info r3_buf = r3.request();
    py::buffer_info r4_buf = r4.request();
    check_vert(r1_buf.format, r1_buf.ndim, r1_buf.shape[0]);
    check_vert(r2_buf.format, r2_buf.ndim, r2_buf.shape[0]);
    check_vert(r3_buf.format, r3_buf.ndim, r3_buf.shape[0]);
    check_vert(r4_buf.format, r4_buf.ndim, r4_buf.shape[0]);

    demcem_ss_ea_nxrwg(static_cast<double *>(r1_buf.ptr), static_cast<double *>(r2_buf.ptr),
                       static_cast<double *>(r3_buf.ptr), static_cast<double *>(r4_buf.ptr),
                       ko, N_theta, N_psi, static_cast<complex<double> *>(result_buf.ptr));
}


void ss_ea_rwg_wrapper(py::array_t<double> r1, py::array_t<double> r2, py::array_t<double> r3,
                       py::array_t<double> r4, const complex<double> ko, const int N_theta,
                       const int N_psi, py::array_t<complex<double>> result)
{
    auto result_buf = result.request();
    if (result_buf.format != py::format_descriptor<complex<double>>::format()){
        throw std::runtime_error("Result must be of dtype complex128");
    }
    if (result_buf.ndim != 1){
        throw std::runtime_error("Result must be a 1D numpy array");
    }
    if (result_buf.shape[0] != 9){
        throw std::runtime_error("Result must have shape (9,)");
    }

    auto r1_buf = r1.request();
    auto r2_buf = r2.request();
    auto r3_buf = r3.request();
    auto r4_buf = r4.request();
    check_vert(r1_buf.format, r1_buf.ndim, r1_buf.shape[0]);
    check_vert(r2_buf.format, r2_buf.ndim, r2_buf.shape[0]);
    check_vert(r3_buf.format, r3_buf.ndim, r3_buf.shape[0]);
    check_vert(r4_buf.format, r4_buf.ndim, r4_buf.shape[0]);

    demcem_ss_ea_rwg(static_cast<double *>(r1_buf.ptr), static_cast<double *>(r2_buf.ptr),
                     static_cast<double *>(r3_buf.ptr), static_cast<double *>(r4_buf.ptr),
                     ko, N_theta, N_psi, static_cast<complex<double> *>(result_buf.ptr));
}


void ss_va_nxrwg_wrapper(py::array_t<double> r1, py::array_t<double> r2, py::array_t<double> r3,
                         py::array_t<double> r4, py::array_t<double> r5, const complex<double> ko,
                         const int N_theta_p, const int N_theta_q, const int N_psi,
                         py::array_t<complex<double>> result)
{
    auto result_buf = result.request();
    if (result_buf.format != py::format_descriptor<complex<double>>::format()){
        throw std::runtime_error("Result must be of dtype complex128");
    }
    if (result_buf.ndim != 1){
        throw std::runtime_error("Result must be a 1D numpy array");
    }
    if (result_buf.shape[0] != 9){
        throw std::runtime_error("Result must have shape (9,)");
    }

    auto r1_buf = r1.request();
    auto r2_buf = r2.request();
    auto r3_buf = r3.request();
    auto r4_buf = r4.request();
    auto r5_buf = r5.request();
    check_vert(r1_buf.format, r1_buf.ndim, r1_buf.shape[0]);
    check_vert(r2_buf.format, r2_buf.ndim, r2_buf.shape[0]);
    check_vert(r3_buf.format, r3_buf.ndim, r3_buf.shape[0]);
    check_vert(r4_buf.format, r4_buf.ndim, r4_buf.shape[0]);
    check_vert(r5_buf.format, r5_buf.ndim, r5_buf.shape[0]);

    demcem_ss_va_nxrwg(static_cast<double *>(r1_buf.ptr), static_cast<double *>(r2_buf.ptr),
                       static_cast<double *>(r3_buf.ptr), static_cast<double *>(r4_buf.ptr),
                       static_cast<double *>(r5_buf.ptr), ko, N_theta_p, N_theta_q, N_psi,
                       static_cast<complex<double> *>(result_buf.ptr));
}


void ss_va_rwg_wrapper(py::array_t<double> r1, py::array_t<double> r2, py::array_t<double> r3,
                       py::array_t<double> r4, py::array_t<double> r5, const complex<double> ko,
                       const int N_theta_p, const int N_theta_q, const int N_psi,
                       py::array_t<complex<double>> result)
{
    auto result_buf = result.request();
    if (result_buf.format != py::format_descriptor<complex<double>>::format()){
        throw std::runtime_error("Result must be of dtype complex128");
    }
    if (result_buf.ndim != 1){
        throw std::runtime_error("Result must be a 1D numpy array");
    }
    if (result_buf.shape[0] != 9){
        throw std::runtime_error("Result must have shape (9,)");
    }

    auto r1_buf = r1.request();
    auto r2_buf = r2.request();
    auto r3_buf = r3.request();
    auto r4_buf = r4.request();
    auto r5_buf = r5.request();
    check_vert(r1_buf.format, r1_buf.ndim, r1_buf.shape[0]);
    check_vert(r2_buf.format, r2_buf.ndim, r2_buf.shape[0]);
    check_vert(r3_buf.format, r3_buf.ndim, r3_buf.shape[0]);
    check_vert(r4_buf.format, r4_buf.ndim, r4_buf.shape[0]);
    check_vert(r5_buf.format, r5_buf.ndim, r5_buf.shape[0]);

    demcem_ss_va_rwg(static_cast<double *>(r1_buf.ptr), static_cast<double *>(r2_buf.ptr),
                     static_cast<double *>(r3_buf.ptr), static_cast<double *>(r4_buf.ptr),
                     static_cast<double *>(r5_buf.ptr), ko, N_theta_p, N_theta_q, N_psi,
                     static_cast<complex<double> *>(result_buf.ptr));
}


void ws_ea_rwg_wrapper(py::array_t<double> r1, py::array_t<double> r2, py::array_t<double> r3,
                       py::array_t<double> r4, const complex<double> ko, const int N_theta,
                       const int N_psi, py::array_t<complex<double>> result)
{
    auto result_buf = result.request();
    if (result_buf.format != py::format_descriptor<complex<double>>::format()){
        throw std::runtime_error("Result must be of dtype complex128");
    }
    if (result_buf.ndim != 1){
        throw std::runtime_error("Result must be a 1D numpy array");
    }
    if (result_buf.shape[0] != 9){
        throw std::runtime_error("Result must have shape (9,)");
    }

    auto r1_buf = r1.request();
    auto r2_buf = r2.request();
    auto r3_buf = r3.request();
    auto r4_buf = r4.request();
    check_vert(r1_buf.format, r1_buf.ndim, r1_buf.shape[0]);
    check_vert(r2_buf.format, r2_buf.ndim, r2_buf.shape[0]);
    check_vert(r3_buf.format, r3_buf.ndim, r3_buf.shape[0]);
    check_vert(r4_buf.format, r4_buf.ndim, r4_buf.shape[0]);

    demcem_ws_ea_rwg(static_cast<double *>(r1_buf.ptr), static_cast<double *>(r2_buf.ptr),
                     static_cast<double *>(r3_buf.ptr), static_cast<double *>(r4_buf.ptr),
                     ko, N_theta, N_psi, static_cast<complex<double> *>(result_buf.ptr));
}


void ws_st_rwg_wrapper(py::array_t<double> r1, py::array_t<double> r2, py::array_t<double> r3,
                       const complex<double> ko, const int Np_1D,
                       py::array_t<complex<double>> result)
{
    auto result_buf = result.request();
    if (result_buf.format != py::format_descriptor<complex<double>>::format()){
        throw std::runtime_error("Result must be of dtype complex128");
    }
    if (result_buf.ndim != 1){
        throw std::runtime_error("Result must be a 1D numpy array");
    }
    if (result_buf.shape[0] != 9){
        throw std::runtime_error("Result must have shape (9,)");
    }

    auto r1_buf = r1.request();
    auto r2_buf = r2.request();
    auto r3_buf = r3.request();
    check_vert(r1_buf.format, r1_buf.ndim, r1_buf.shape[0]);
    check_vert(r2_buf.format, r2_buf.ndim, r2_buf.shape[0]);
    check_vert(r3_buf.format, r3_buf.ndim, r3_buf.shape[0]);

    demcem_ws_st_rwg(static_cast<double *>(r1_buf.ptr), static_cast<double *>(r2_buf.ptr),
                     static_cast<double *>(r3_buf.ptr), ko, Np_1D,
                     static_cast<complex<double> *>(result_buf.ptr));
}


void ws_va_rwg_wrapper(py::array_t<double> r1, py::array_t<double> r2, py::array_t<double> r3,
                       py::array_t<double> r4, py::array_t<double> r5, const complex<double> ko,
                       const int N_theta_p, const int N_theta_q, const int N_psi,
                       py::array_t<complex<double>> result)
{
    auto result_buf = result.request();
    if (result_buf.format != py::format_descriptor<complex<double>>::format()){
        throw std::runtime_error("Result must be of dtype complex128");
    }
    if (result_buf.ndim != 1){
        throw std::runtime_error("Result must be a 1D numpy array");
    }
    if (result_buf.shape[0] != 9){
        throw std::runtime_error("Result must have shape (9,)");
    }

    auto r1_buf = r1.request();
    auto r2_buf = r2.request();
    auto r3_buf = r3.request();
    auto r4_buf = r4.request();
    auto r5_buf = r5.request();
    check_vert(r1_buf.format, r1_buf.ndim, r1_buf.shape[0]);
    check_vert(r2_buf.format, r2_buf.ndim, r2_buf.shape[0]);
    check_vert(r3_buf.format, r3_buf.ndim, r3_buf.shape[0]);
    check_vert(r4_buf.format, r4_buf.ndim, r4_buf.shape[0]);
    check_vert(r5_buf.format, r5_buf.ndim, r5_buf.shape[0]);

    demcem_ws_va_rwg(static_cast<double *>(r1_buf.ptr), static_cast<double *>(r2_buf.ptr),
                     static_cast<double *>(r3_buf.ptr), static_cast<double *>(r4_buf.ptr),
                     static_cast<double *>(r5_buf.ptr), ko, N_theta_p, N_theta_q, N_psi,
                     static_cast<complex<double> *>(result_buf.ptr));
}


PYBIND11_MODULE(demcem_bindings, m)
{
    m.doc() = "Binding to DEMCEM C++ package for evaluation of singular integrals.";

    m.def("ss_ea_nxrwg", &ss_ea_nxrwg_wrapper,
          "Compute strongly singular integral with nxRWG testing function for edge adjacent triangles."
          " Result array is modified in-place.",
          py::arg("r1"), py::arg("r2"), py::arg("r3"), py::arg("r4"), py::arg("ko"),
          py::arg("N_theta"), py::arg("N_psi"), py::arg("result"));

    m.def("ss_ea_rwg", &ss_ea_rwg_wrapper,
          "Compute strongly singular integral with RWG testing function for edge adjacent triangles."
          " Result array is modified in-place.",
          py::arg("r1"), py::arg("r2"), py::arg("r3"), py::arg("r4"), py::arg("ko"),
          py::arg("N_theta"), py::arg("N_psi"), py::arg("result"));

    m.def("ss_va_nxrwg", &ss_va_nxrwg_wrapper,
          "Compute strongly singular integral with nxRWG testing function for vertex adjacent triangles."
          " Result array is modified in-place.",
          py::arg("r1"), py::arg("r2"), py::arg("r3"), py::arg("r4"), py::arg("r5"),
          py::arg("ko"), py::arg("N_theta_p"), py::arg("N_theta_q"), py::arg("N_psi"),
          py::arg("result"));

    m.def("ss_va_rwg", &ss_va_rwg_wrapper,
          "Compute strongly singular integral with RWG testing function for vertex adjacent triangles."
          " Result array is modified in-place.",
          py::arg("r1"), py::arg("r2"), py::arg("r3"), py::arg("r4"), py::arg("r5"),
          py::arg("ko"), py::arg("N_theta_p"), py::arg("N_theta_q"), py::arg("N_psi"),
          py::arg("result"));

    m.def("ws_ea_rwg", &ws_ea_rwg_wrapper,
          "Compute weakly singular integral with RWG testing function for edge adjacent triangles."
          " Result array is modified in-place.",
          py::arg("r1"), py::arg("r2"), py::arg("r3"), py::arg("r4"),  py::arg("ko"),
          py::arg("N_theta"), py::arg("N_psi"), py::arg("result"));

    m.def("ws_st_rwg", &ws_st_rwg_wrapper,
          "Compute weakly singular integral with RWG testing function for self-term."
          " Result array is modified in-place.",
          py::arg("r1"), py::arg("r2"), py::arg("r3"), py::arg("ko"),
          py::arg("Np_1D"), py::arg("result"));

    m.def("ws_va_rwg", &ws_va_rwg_wrapper,
          "Compute weakly singular integral with RWG testing function for vertex adjacent triangles."
          " Result array is modified in-place.",
          py::arg("r1"), py::arg("r2"), py::arg("r3"), py::arg("r4"), py::arg("r5"),
          py::arg("ko"), py::arg("N_theta_p"), py::arg("N_theta_q"), py::arg("N_psi"),
          py::arg("result"));
}
