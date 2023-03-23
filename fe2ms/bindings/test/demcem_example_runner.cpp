/*
 * Program for running DEMCEM directly to obtain reference results for the pybind11 version.
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
#include <iostream>
#include <fstream>
#include <complex>

#include "demcem_ss_ea_nxrwg.h"
#include "demcem_ss_ea_rwg.h"
#include "demcem_ss_va_nxrwg.h"
#include "demcem_ss_va_rwg.h"
#include "demcem_ws_ea_rwg.h"
#include "demcem_ws_st_rwg.h"
#include "demcem_ws_va_rwg.h"


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


// argv[1] is the number of integration points N
int main(int argc, char *argv[])
{
    int N;
    if(argc < 2)
    {
        std::cout << "Number of integration points not set, using N=5\n";
        N = 5;
    }
    else
    {
        N = atoi(argv[1]);
        if(N == 0)
        {
            std::cout << "Invalid command line argument, using N=5\n";
            N = 5;
        }
    }

    complex<double> I_DE[9];
    double r1[3];
    double r2[3];
    double r3[3];
    double r4[3];
    double r5[3];
    double ko;
    
    ////////////////////////////////////////////////////////////////////////
    //    Run equivalent of all examples and write results to file
    ////////////////////////////////////////////////////////////////////////

    // SS EA nxRWG & RWG

   r1[0] = 0.0;
   r1[1] = 0.0;
   r1[2] = 0.0;

   r2[0] = 0.0;
   r2[1] = 0.1;
   r2[2] = 0.0;

   r3[0] = 0.0;
   r3[1] = 0.0;
   r3[2] = 0.1;

   r4[0] = 0.1;
   r4[1] = 0.0;
   r4[2] = 0.0;
   ko = 2.0 * M_PI;

    demcem_ss_ea_nxrwg ( r1, r2, r3, r4, ko, N, N, I_DE );

    ofstream out_file;
    out_file.open("demcem_ref_ss_ea_nxrwg.txt");
    for(int i = 0; i < 9; i++)
    {
        out_file << real(I_DE[i]) << " " << imag(I_DE[i]) << "\n";
    }
    out_file.close();
    out_file.clear();

    demcem_ss_ea_rwg ( r1, r2, r3, r4, ko, N, N, I_DE );

    out_file.open("demcem_ref_ss_ea_rwg.txt");
    for(int i = 0; i < 9; i++)
    {
        out_file << real(I_DE[i]) << " " << imag(I_DE[i]) << "\n";
    }
    out_file.close();
    out_file.clear();


    // SS VA nxRWG & RWG

    r1[0] = 0.1;
    r1[1] = 0.1;
    r1[2] = 0.1;

    r2[0] = 0.2;
    r2[1] = 0.1;
    r2[2] = 0.1;

    r3[0] = 0.1;
    r3[1] = 0.2;
    r3[2] = 0.1;

    r4[0] = 0.0;
    r4[1] = 0.1;
    r4[2] = 0.2;

    r5[0] = 0.0;
    r5[1] = 0.2;
    r5[2] = 0.2;
    ko = 2.0 * M_PI;

    demcem_ss_va_nxrwg ( r1, r2, r3, r4, r5, ko, N, N, N, I_DE );

    out_file.open("demcem_ref_ss_va_nxrwg.txt");
    for(int i = 0; i < 9; i++)
    {
        out_file << real(I_DE[i]) << " " << imag(I_DE[i]) << "\n";
    }
    out_file.close();
    out_file.clear();

    demcem_ss_va_rwg ( r1, r2, r3, r4, r5, ko, N, N, N, I_DE );

    out_file.open("demcem_ref_ss_va_rwg.txt");
    for(int i = 0; i < 9; i++)
    {
        out_file << real(I_DE[i]) << " " << imag(I_DE[i]) << "\n";
    }
    out_file.close();
    out_file.clear();


    // WS EA RWG

   r1[0] = 0.0;
   r1[1] = 0.0;
   r1[2] = 0.0;

   r2[0] = 0.5;
   r2[1] = 0.0;
   r2[2] = 0.0;

   r3[0] = 0.0;
   r3[1] = 0.0;
   r3[2] = 0.5;

   r4[0] = 0.0;
   r4[1] = 0.5;
   r4[2] = 0.0;
   ko = 1.0;

    demcem_ws_ea_rwg ( r1, r2, r3, r4, ko, N, N, I_DE );

    out_file.open("demcem_ref_ws_ea_rwg.txt");
    for(int i = 0; i < 9; i++)
    {
        out_file << real(I_DE[i]) << " " << imag(I_DE[i]) << "\n";
    }
    out_file.close();
    out_file.clear();

    // WS ST RWG

   r1[0] = 0.0;
   r1[1] = 0.0;
   r1[2] = 0.5;

   r2[0] = 0.0;
   r2[1] = 0.0;
   r2[2] = 0.0;

   r3[0] = 0.5;
   r3[1] = 0.0;
   r3[2] = 0.0;
   ko = 2.0 * M_PI;

    demcem_ws_st_rwg ( r1, r2, r3, ko, N, I_DE );

    out_file.open("demcem_ref_ws_st_rwg.txt");
    for(int i = 0; i < 9; i++)
    {
        out_file << real(I_DE[i]) << " " << imag(I_DE[i]) << "\n";
    }
    out_file.close();
    out_file.clear();


    // WS VA RWG

   r1[0] = 1.0;
   r1[1] = 1.0;
   r1[2] = 1.0;

   r2[0] = 2.0;
   r2[1] = 1.0;
   r2[2] = 1.0;

   r3[0] = 1.0;
   r3[1] = 2.0;
   r3[2] = 1.0;

   r4[0] = 0.0;
   r4[1] = 1.0;
   r4[2] = 1.0;

   r5[0] = 0.0;
   r5[1] = 2.0;
   r5[2] = 1.0;
   ko = 2.0 * M_PI;

    demcem_ws_va_rwg ( r1, r2, r3, r4, r5, ko, N, N, N, I_DE );

    out_file.open("demcem_ref_ws_va_rwg.txt");
    for(int i = 0; i < 9; i++)
    {
        out_file << real(I_DE[i]) << " " << imag(I_DE[i]) << "\n";
    }
    out_file.close();
    out_file.clear();

    return 0;

}