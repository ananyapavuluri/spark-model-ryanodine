
//CUDA sticky2011 test --  Ananya Pavuluri

/* Monte - Carlo model of Ca spark from single cluster of RyRs
Parameters specified in: Ramay et. al, Cardiovascular Research, 2011
Model from Sobie et. al, Biophysics Journal, 2002 */

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <time.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define trials 100
using namespace std; // NOTE: When the number of trials is changed, the number of blocks in kernel launch changes as well.

__global__ void simulation(double F, double V_ss, double V_JSR, double tau_efflux, double tau_refill, double D_ryr, double EJequiv,
	double kr_minus, double kr_plus_max, double kr_plus_ryanodine, double Km_r_max, double Km_r_ryanodine, double alpha_r, int hill, int N_RyR, int N_RyR_ryanodine, double kcoup, double CSQ, double KCSQ,
    	double dt, const double dt_record, const double interval, const double timeafter, double t_end, int iterations, int outputs,
	double *devCassall, double *devCaJSRall, double *devIrelall, double *devNopenall, double *devNopenryanodine);


__global__ void simulation(double F, double V_ss, double V_JSR, double tau_efflux, double tau_refill, double D_ryr, double EJequiv,
	double kr_minus, double kr_plus_max, double kr_plus_ryanodine, double Km_r_max, double Km_r_ryanodine, double alpha_r, int hill, int N_RyR,
	int N_RyR_ryanodine, double kcoup, double CSQ, double KCSQ, double dt, const double dt_record, const double interval,
	const double timeafter, double t_end, int iterations, int outputs, double *devCassall, double *devCaJSRall, double *devIrelall, double *devNopenall, double *devNopenryanodine)
{
	// initializing CUDA random number generator
	curandState rndState;
	curand_init(clock64(), 13, 0, &rndState);

	// Subspace buffers 

	// Total [calmodulin], Total [SR membrane buffer],  Total [SL membrane buffer]
	double bt[3] = { 24, 47, 900 }; // uM
									// On rates for calmodulin, SR membrane, SL membrane
	double k_on[3] = { 100, 115, 115 };
	// Off rates for calmodulin, SR membrane, SL membrane
	double k_off[3] = { 38, 100, 1000 };
	// converting from s^-1 to ms^-1
	for (int n = 0; n < 3; n++) {
		k_on[n] *= 0.0010;
		k_off[n] *= 0.0010;
	}


	// Fixed ionic concentrations
	double Ca_myo = 0.1;        // bulk myoplasm [Ca2+]
	double Ca_NSR = 1000;		  // NSR [Ca2+]
	double Ca_ss = 0.1;         // subspace [Ca2+]
	double Ca_JSR = 1000;       // JSR [Ca2+]

								// Making array for buffers
	double b[3];

	double J_ryr = 0;
	double kr_plus_max_ryanodine = 23052865.2;
	// ---------------------        SIMULATION       --------------------- // 

	// INITIAL CONDITIONS
	Ca_ss = Ca_myo;
	Ca_JSR = Ca_NSR;
	double nopen_normal = 0;
	double nopen_ryanodine = 0;
	double nopen = nopen_normal + nopen_ryanodine;

	// buffers
	for (int i = 0; i < 3; i++) {
		b[i] = (bt[i] * (k_off[i] / k_on[i])) / (k_off[i] / k_on[i] + Ca_ss);
	}

	int writedex = 0;
	double tlast = -1 * dt;
	int J_d = 0;
	bool neverspark = true;
	double time = 0.0;
	
	//printf("before simulation");
	
	for (int j = 0; j < iterations; j++) {


		/*if (time >= interval && tlast < interval) {
			nopen = nopen + 5;
			printf("in\n");
		}
		/*else if (time >= interval + 10 && nopen < 1 && neverspark) {
			break;
		} */

		double nclosed = N_RyR - nopen;
		double nclosed_ryanodine = N_RyR_ryanodine - nopen_ryanodine;
		double nclosed_normal = (N_RyR - N_RyR_ryanodine) - nopen_normal;


		// Fluxes and currents	
		J_ryr = nopen * D_ryr * (Ca_JSR - Ca_ss) / V_ss; // uM/ms
		double I_ryr = 1e6 * J_ryr * 2 * F * V_ss; 	 // pA
		double J_efflux = (Ca_myo - Ca_ss) / tau_efflux;
		double J_refill = (Ca_NSR - Ca_JSR) / tau_refill;

		// Buffers
		double db_dt[3];
		double J_buff = 0;
		for (int i = 0; i < 3; i++) {
			db_dt[i] = -1 * k_on[i] * b[i] * Ca_ss + k_off[i] * (bt[i] - b[i]);
			J_buff += db_dt[i];
		}

		double denom = pow(KCSQ + Ca_JSR, 2);
		double B_JSR = pow((1 + CSQ * KCSQ / denom), -1);

		// Writing arrays after the fluxes are calculated, before integration
		// and state switching

		if (j % (iterations / (outputs - 1)) == 0) {
			dt = 0.0000100; // the value of dt changes to 9.99999974737875e-06 
							// within this conditional for an unknown reason.
							// here, it is assigned 1e-5 again to prevent this.
							//printf("block: %d\n", blockIdx.x);
			//printf("nopen: %lf\n", nopen);
			//printf("iteration: %d\n", j);
			int idx = blockDim.x * blockIdx.x + threadIdx.x;
			devIrelall[idx * outputs + writedex] = I_ryr;
			devNopenall[idx * outputs + writedex] = nopen;
			devNopenryanodine[idx * outputs + writedex] = nopen_ryanodine;
			devCassall[idx * outputs + writedex] = Ca_ss;
			devCaJSRall[idx * outputs + writedex] = Ca_JSR;
			writedex = writedex + 1;
		}

		double Km_r = Km_r_max - alpha_r * Ca_JSR;
		double pow1 = pow(Ca_ss, hill);
		double pow2 = pow(Km_r, hill);
		double kr_plus = kr_plus_max * pow1 / (pow1 + pow2);
		kr_plus_ryanodine = kr_plus_max_ryanodine * pow(0.1, hill) / (pow(0.1, hill) + pow(Km_r_ryanodine, hill));

		// Stochastic variables
		double  pincrease = dt * nclosed * kr_plus * pow(kcoup, 2 * nopen + 1 - N_RyR);
		//printf("pinc: %lf\n", pincrease);
		double  pdecrease = dt * nopen * kr_minus * pow(kcoup, 2 * nclosed + 1 - N_RyR);
		double	pincrease_ryanodine = dt*nclosed_ryanodine*kr_plus_ryanodine*pow(kcoup, (2 * nopen + 1 - N_RyR));
		//printf("pinc: %lf\n", pincrease_ryanodine);

		// opening of a free RyR
		if (curand_uniform(&rndState) < pincrease) {
			if ((nopen + 1) <= N_RyR) {
				nopen_normal = nopen_normal + 1;
			}
		}

		nopen += nopen_normal;

		// opening for ryanodine bound RyR
		if (curand_uniform(&rndState) < pincrease_ryanodine) {
			if ((nopen + 1) <= N_RyR) {
				nopen_ryanodine = nopen_ryanodine + 1;
			}
		}
		nopen += nopen_ryanodine;
		// closing free RyR
		if (nopen_normal != 0) {
			if (curand_uniform(&rndState) < pdecrease) {
				nopen_normal = nopen_normal - 1;
			}
		}
		// closing ryanodine-bound RyR
		if (nopen_ryanodine != 0) {
			if (curand_uniform(&rndState) < pdecrease) {
				nopen_ryanodine = nopen_ryanodine - 1;
			}
		}

		nopen = nopen_normal + nopen_ryanodine;

		if (nopen >= 5) {
			neverspark = false;
		}

		// Subspace [Ca2+] and JSR [Ca2+] derivatives
		double dCass_dt = J_efflux + J_d + J_ryr + J_buff;
		double dCaJSR_dt = B_JSR * (J_refill - J_ryr * V_ss / V_JSR);

		Ca_ss = Ca_ss + dt * dCass_dt;
		Ca_JSR = Ca_JSR + dt * dCaJSR_dt;

		// updating buffers
		for (int i = 0; i < 3; i++) {
			b[i] += dt*db_dt[i];
		}

		tlast = time;
		time += dt;
		// accounting for precision problems with floating point numbers.
		// ensures that time is accurate to 5 decimal places.
		time = round(time * 100000) / 100000;
		
	}

	// Write values at last time point t_end after the loop terminates

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	devCassall[index * outputs + writedex] = Ca_ss;
	devCaJSRall[index * outputs + writedex] = Ca_JSR;
	devIrelall[index * outputs + writedex] = 1e6 * J_ryr * 2 * F * V_ss;
	devNopenall[index * outputs + writedex] = nopen;
	devNopenryanodine[index * outputs + writedex] = nopen_ryanodine;

	printf("kernel end");
}




// ---------------HOST MEMORY ---------------
int main(void)

{

	// Faraday's Constant. Used to convert flux to current
	double F = 96.485; // C/mmol

					   // Geometrical parameters
	double V_ss = 1.0000e-12;
	double V_JSR = 1.6000e-12;

	// Time constants, in ms
	// time constant for ...
	double tau_efflux = 1.78e-3;
	// time constant for NSR to JSR refilling
	double tau_refill = 6.5;

	//Coupling energy between RyRs
	double EJequiv = 0.1;

	// RyR permeability constant
	double D_ryr = 2.2e-12;

	// RyR gating parameters
	double kr_minus, kr_plus_max, Km_r_max, alpha_r, hill, kr_plus_ryanodine, Km_r_ryanodine;
	kr_minus = 0.48;        // max close rate, ms^-1
	kr_plus_max = 30.0;         // max open rate, ms^-1
	kr_plus_ryanodine = 0;		// open rate when ryanodine is bound
	Km_r_max = 19.87;           // sensitivity of opening to subspace Ca, uM
	alpha_r = 1.0e-3;           // luminal dependence factor 
	Km_r_ryanodine = Km_r_max - alpha_r * 0.1;			// sensitivity of opening with ryanodine
	hill = 4;                   // exponent

	int N_RyR = 28; // # of RyR channels in a cluster
	int N_RyR_ryanodine = 1; // # of ryanodine-bound RyRs

							 // Coupling rate
	double kcoup = exp(2 * EJequiv / (N_RyR - 1));


	// JSR buffer calsequestrin
	double CSQ = 30e3; // uM  // total [CSQ]
	double KCSQ = 630; // uM  // Ca dissociation constant 

	// time steps
	double dt = 0.0000100;
	const double dt_record = 0.1000000;

	// Open single RyR at interval, run for 'timeafter' ms
	const double interval = 1;
	const double timeafter = 2500.0; // change this to change amount of data.  

									//Initializing arrays to hold results
	double t_end = interval + timeafter;

	int iterations = nearbyint(t_end / dt);
	int outputs = nearbyint(t_end / dt_record) + 1;

	int plottime_size = (int)outputs * dt_record * 10;

	// ALLOCATING HEAP MEMORY FOR DYNAMIC ARRAYS

	// times for the plot
	double *plottime = new double[plottime_size];
	double timepoint = 0;
	for (int i = 0; i < plottime_size; i++) {
		plottime[i] = timepoint;
		timepoint = round((timepoint + dt_record) * 10) / 10;
	}
	// "host" arrays are flattened 1d arrays that will temporarily store:
	// JSR [Ca2+], subspace [Ca2+], number of RyRs open, and currents, respectively
	double *hostCaJSRall = new double[outputs * trials];
	double *devCaJSRall = new double[outputs * trials];

	// testing cudaMalloc()
	if (cudaSuccess != cudaMalloc((void**)&devCaJSRall, outputs * trials * sizeof(double))) {
		cout << "Malloc fail" << endl;
	}
	double *hostCassall = new double[outputs* trials];
	double *devCassall = new double[outputs * trials];
	cudaMalloc((void**)&devCassall, outputs * trials * sizeof(double));
	double *hostNopenall = new double[outputs * trials];
	double *devNopenall = new double[outputs * trials];
	cudaMalloc((void**)&devNopenall, outputs * trials * sizeof(double));
	double *hostIrelall = new double[outputs * trials];
	double *devIrelall = new double[outputs * trials];
	cudaMalloc((void**)&devIrelall, outputs * trials * sizeof(double));
	double *hostNopenryanodine = new double[outputs * trials];
	double *devNopenryanodine = new double[outputs * trials];
	cudaMalloc((void**)&devNopenryanodine, outputs * trials * sizeof(double));


	// testing cudaMemcpy()
	if (cudaSuccess != cudaMemcpy(devCaJSRall, hostCaJSRall, outputs * trials * sizeof(double), cudaMemcpyHostToDevice)) {
		cout << "memcpy fail" << endl;
	}
	cudaMemcpy(devCassall, hostCassall, outputs * trials * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devNopenall, hostNopenall, outputs * trials * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devIrelall, hostIrelall, outputs * trials * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devNopenryanodine, hostNopenryanodine, outputs * trials * sizeof(double), cudaMemcpyHostToDevice);

	cout << "Starting simulation" << endl;
	cout << outputs * trials << endl;
	simulation << <trials, 1 >> > (F, V_ss, V_JSR, tau_efflux, tau_refill, D_ryr, EJequiv, kr_minus, kr_plus_max, kr_plus_ryanodine, Km_r_max, Km_r_ryanodine, alpha_r, hill, N_RyR, N_RyR_ryanodine, kcoup, CSQ, KCSQ,
		dt, dt_record, interval, timeafter, t_end, iterations, outputs, devCassall, devCaJSRall, devIrelall, devNopenall, devNopenryanodine);
	if (cudaDeviceSynchronize() != cudaSuccess) {
		cout << "sync fail" << endl;
	}
	cout << "End of simulation" << endl;
	cudaMemcpy(hostCassall, devCassall, outputs * trials * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(devCassall);
	cudaMemcpy(hostCaJSRall, devCaJSRall, outputs * trials * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(devCaJSRall);
	cudaMemcpy(hostNopenall, devNopenall, outputs * trials * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(devNopenall);
	cudaMemcpy(hostIrelall, devIrelall, outputs * trials * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(devIrelall);
	cudaMemcpy(hostNopenryanodine, devNopenryanodine, outputs * trials * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(devNopenryanodine);

	// storing information from flattened 1d arrays into 2d arrays

	// Array for [Ca2+] in the JSR
	double **Ca_JSR_all = new double *[outputs];
	for (int i = 0; i < outputs; i++) {
		Ca_JSR_all[i] = new double[trials];
	}

	// Array for subspace [Ca2+]
	double **Ca_ss_all = new double *[outputs];
	for (int i = 0; i < outputs; i++) {
		Ca_ss_all[i] = new double[trials];
	}

	// Array for current released from RyR
	double **Irel_all = new double *[outputs];
	for (int i = 0; i < outputs; i++) {
		Irel_all[i] = new double[trials];
	}

	// Array for number of receptors open
	double **Nopen_all = new double *[outputs];
	for (int i = 0; i < outputs; i++) {
		Nopen_all[i] = new double[trials];
	}

	// Array for number of receptors open with ryanodine
	double **Nopen_ryanodine = new double *[outputs];
	for (int i = 0; i < outputs; i++) {
		Nopen_ryanodine[i] = new double[trials];
	}

	for (int x = 0; x < outputs; x++) {
		for (int y = 0; y < trials; y++) {
			Ca_JSR_all[x][y] = 0;
			Ca_ss_all[x][y] = 0;
			Irel_all[x][y] = 0;
			Nopen_all[x][y] = 0;
			Nopen_ryanodine[x][y] = 0;

		}
	}

	int count = 0;
	for (int x = 0; x < trials; x++) {
		for (int y = 0; y < outputs; y++) {
			Ca_JSR_all[y][x] = hostCaJSRall[count];
			Ca_ss_all[y][x] = hostCassall[count];
			Nopen_all[y][x] = hostNopenall[count];
			Irel_all[y][x] = hostIrelall[count];
			Nopen_ryanodine[y][x] = hostNopenryanodine[count];
			count++;
		}
	}

	cout << "Begin writing output" << endl;

	ofstream Nopen;
	Nopen.open("N_open.csv");
	while (Nopen.is_open()) {
		for (int i = 0; i < outputs; i++) {
			Nopen << Nopen_all[i][0];
			for (int j = 1; j < trials; j++) {
				Nopen << "," << Nopen_all[i][j];
			}
			Nopen << endl;
		}
		break;
	}
	Nopen.close();
	ofstream Irel;
	Irel.open("Irel.csv");
	while (Irel.is_open()) {
		for (int i = 0; i < outputs; i++) {
			Irel << Irel_all[i][0];
			for (int j = 1; j < trials; j++) {
				Irel << "," << Irel_all[i][j];
			}
			Irel << endl;
		}
		break;
	}
	Irel.close();
	ofstream Cads;
	Cads.open("Ca_ss.csv");
	while (Cads.is_open()) {
		for (int i = 0; i < outputs; i++) {
			Cads << Ca_ss_all[i][0];
			for (int j = 1; j < trials; j++) {
				Cads << "," << Ca_ss_all[i][j];
			}
			Cads << endl;
		}
		break;
	}
	Cads.close();
	ofstream CaJSR;
	CaJSR.open("CaJSR.csv");
	while (CaJSR.is_open()) {
		for (int i = 0; i < outputs; i++) {
			CaJSR << Ca_JSR_all[i][0];
			for (int j = 1; j < trials; j++) {
				CaJSR << "," << Ca_JSR_all[i][j];
			}
			CaJSR << endl;
		}
		break;
	}
	ofstream Nopen_ry;
	Nopen_ry.open("Nopenryanodine.csv");
	while (Nopen_ry.is_open()) {
		for (int i = 0; i < outputs; i++) {
			Nopen_ry << Nopen_ryanodine[i][0];
			for (int j = 1; j < trials; j++) {
				Nopen_ry << "," << Nopen_ryanodine[i][j];
			}
			Nopen_ry << endl;
		}
		break;
	}
	Nopen_ry.close();

	ofstream plot_time;
	plot_time.open("plottime.csv");
	while (plot_time.is_open()) {
		for (int i = 0; i < outputs; i++) {
			plot_time << plottime[i];
			if (i != (outputs - 1))
				plot_time << ",";
		}
		break;
	}
	plot_time.close();
	// Recycling memory
	for (int k = 0; k < trials; k++) {
		delete[] Ca_ss_all[k];
		delete[] Ca_JSR_all[k];
		delete[] Nopen_all[k];
		delete[] Nopen_ryanodine[k];
		delete[] Irel_all[k];
	}

	delete[] hostCaJSRall;
	delete[] hostCassall;
	delete[] hostNopenall;
	delete[] hostIrelall;
	delete[] hostNopenryanodine;

	cout << "End of program." << endl;

	return 0;
}

