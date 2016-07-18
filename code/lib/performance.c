/**
 *      _________   _____________________  ____  ______
 *     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
 *    / /_  / /| | \__ \ / / / /   / / / / / / / __/
 *   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
 *  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
 *
 *  http://www.inf.ethz.ch/personal/markusp/teaching/
 *  How to Write Fast Numerical Code 263-2300 - ETH Zurich
 *  Copyright (C) 2015  Alen Stojanov      (astojanov@inf.ethz.ch)
 *                      Daniele Spampinato (daniele.spampinato@inf.ethz.ch)
 *                      Singh Gagandeep    (gsingh@inf.ethz.ch)
 *	                    Markus Pueschel    (pueschel@inf.ethz.ch)
 *	Copyright (C) 2013  Georg Ofenbeck     (ofenbeck@inf.ethz.ch)
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see http://www.gnu.org/licenses/.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "CPUInfo.h"
#include "performance.h"
#include "perf.h"

unsigned int MHz = 1800;

CPUINFO cpu_info;

/* Prints a message when a function fails validation */
void invalid_msg (unsigned int size, char *desc) {
	printf("Invalid result for %s (size %d)\n", desc, size);
}

/*
 * get_cycles_time - Get the number of cycles of f(argp)
 * using RDTSC and convert it into time. Return the average of n runs.
 * (Adapted to work with 3 argument matrix funcs)
 */
double get_cycles_time (comp_func f, struct TestParams* tparams, int iters) {
	myInt64 values[MAX_NUM_MEASUREMENTS];
	double t, result = 0;
	int i;
	measurement_init ();
	for (i = 0; i < iters; i++) {
		measurement_start();
		f(tparams->wx, tparams->wy, tparams->im1, tparams->im2, tparams->params, tparams->match_x, tparams->match_y, tparams->match_z);
		measurement_stop();
	}
	measurement_finish(values);
	for (i = 0; i < iters; i++) {
		t = (double) values[i] / (double) iters;
		result += t;
	}
	return result;
}

/**
 * Gets the performance of a given function with given number of iterations
 * Result is in MFLOPs
 */
double get_perf_score(comp_func f, struct TestParams* tparams, unsigned int iters) {
	double cycles, seconds;
	cycles  = get_cycles_time(f, tparams, iters);

	seconds = cycles / (MHz * 1e6);
	//ops     = (2 * pow(size, 2));
	//mflops  = ops / (seconds * 1e6);
	//fpc     = ops / cycles;

	//printf("Size %5d: %18.3f cycles, %7.4f F/C | estimated: %7.4f sec, %8.3f MFLOPs\n",
	//		size, cycles, fpc, seconds, mflops);
	printf("%18.3f cycles, %7.4f sec.\n", cycles, seconds);
	return cycles;
}

/*
 * Checks the given function for validity. If valid, then computes and
 * reports and returns its performance in MFLOPs
 */
double perf_test(comp_func f, char *desc, struct TestParams* tparams,
		unsigned int test_iter, unsigned int num_tests)
{
	double results_sum, avg;
	static int first = 1;
	unsigned int i; // test_size;

	//Seed the random number generator once and only once
	if (first) {
		srand(time(0));
		first = 0;
	}
	printf("%s:\n", desc);
	results_sum = 0;
	for (i = 0; i < num_tests; i++) {
		results_sum += get_perf_score(f, tparams, test_iter);
	}
	avg = results_sum / num_tests;
	printf("Average: %.4f cycles\n\n", avg);
	return avg;
}

void get_CPU_info () {
	static int obtained = 0;
	if (obtained == 0) {
		GetCPUInfo(&cpu_info, 1);
		obtained = 1;
	}
}

void get_CPU_freq () {
	get_CPU_info ();
	MHz = cpu_info.clockspeed;
}

void print_CPU_info () {
	get_CPU_info ();
	CPUINFO* pCPUInfo = &cpu_info;
	char buf[65536];
	char* pBuf = buf;
	pBuf += sprintf( pBuf, "Family: %u\r\n"
			"FPU:  %u\r\n"
			"Model: %u\r\n"
			"Stepping: %u\r\n"
			"Type: %u\r\n"
			"AMD Family: %u\r\n"
			"AMD Model: %u\r\n"
			"AMD Stepping: %u\r\n"
			"Features: %08X\r\n"
			"Extended Features: %08X\r\n"
			"Extended Intel Features: %08X\r\n"
			"Extended AMD Features: %08X\r\n"
			"Brand index: %u\r\n"
			"CLFLUSH Line Size: %u\r\n"
			"Number of Logical Processors: %u\r\n"
			"APIC ID: %u\r\n"
			"Clockspeed: %u MHz\r\n"
			"Vendor ID: %s\r\n"
			"Processor Brand String: %s\r\n"
			"Serial number: %llu\r\n"
			"Maximum CPUID input value: %08X\r\n"
			"Maximum extended CPUID input value: %08X\r\n",
			pCPUInfo->family,
			pCPUInfo->fpu,
			pCPUInfo->model,
			pCPUInfo->stepping,
			pCPUInfo->type,
			pCPUInfo->amdFamily,
			pCPUInfo->amdModel,
			pCPUInfo->amdStepping,
			pCPUInfo->features,
			pCPUInfo->extFeatures,
			pCPUInfo->extIntelFeatures,
			pCPUInfo->extAMDFeatures,
			pCPUInfo->brandIndex,
			pCPUInfo->CLFLUSH,
			pCPUInfo->logicalCPUs,
			pCPUInfo->APICID,
			pCPUInfo->clockspeed,
			pCPUInfo->vendorID,
			pCPUInfo->brandString,
			pCPUInfo->serial,
			pCPUInfo->maxInput,
			pCPUInfo->maxExtInput );

	// Print cache descriptors
	pBuf += sprintf( pBuf, "Cache descriptors:\r\n" );

	unsigned char* desc = pCPUInfo->cacheDescriptors;
	while (*desc)
	{
		pBuf += sprintf( pBuf, " %02X: %s\r\n", *desc, descriptorStrings[*desc] );
		desc++;
	}

	pBuf += sprintf( pBuf, "\r\n" );

	pBuf += sprintf( pBuf, "Type string: %s\r\n", typeStrings[pCPUInfo->type] );

	pBuf += sprintf( pBuf, "Brand string: %s\r\n", brandStrings[pCPUInfo->brandIndex] );

	// Print some features
	pBuf += sprintf( pBuf, "FPU: %u\r\n", HasFPU( pCPUInfo ) );
	pBuf += sprintf( pBuf, "MMX: %u\r\n", HasMMX( pCPUInfo ) );
	pBuf += sprintf( pBuf, "Extended MMX: %u\r\n", HasMMXExt( pCPUInfo ) );
	pBuf += sprintf( pBuf, "3DNow!: %u\r\n", Has3DNow( pCPUInfo ) );
	pBuf += sprintf( pBuf, "Extended 3DNow!: %u\r\n", Has3DNowExt( pCPUInfo ) );
	pBuf += sprintf( pBuf, "SSE: %u\r\n", HasSSE( pCPUInfo ) );
	pBuf += sprintf( pBuf, "SSE2: %u\r\n", HasSSE2( pCPUInfo ) );
	pBuf += sprintf( pBuf, "SSE3: %u\r\n", HasSSE3( pCPUInfo ) );
	pBuf += sprintf( pBuf, "SSSE3: %u\r\n", HasSSSE3( pCPUInfo ) );
	pBuf += sprintf( pBuf, "SSE4.1: %u\r\n", HasSSE4_1( pCPUInfo ) );
	pBuf += sprintf( pBuf, "SSE4.2: %u\r\n", HasSSE4_2( pCPUInfo ) );
	pBuf += sprintf( pBuf, "64-bit: %u\r\n", Is64Bit( pCPUInfo ) );
	pBuf += sprintf( pBuf, "HyperThreading: %u\r\n", HasHTT( pCPUInfo ) );
	pBuf += sprintf( pBuf, "Serial number: %u\r\n", HasSerial( pCPUInfo ) );

	// AMD Cache info
	pBuf += sprintf( pBuf, "L1 code TLB (large) entries: %u\r\n"
			"L1 code TLB (large) associativity: %u\r\n"
			"L1 data TLB (large) entries: %u\r\n"
			"L1 data TLB (large) associativity: %u\r\n"

			"L1 code TLB entries: %u\r\n"
			"L1 code TLB associativity: %u\r\n"
			"L1 data TLB entries: %u\r\n"
			"L1 data TLB associativity: %u\r\n"

			"L2 data TLB (large) entries: %u\r\n"
			"L2 data TLB (large) associativity: %u\r\n"
			"L2 data TLB entries: %u\r\n"
			"L2 data TLB associativity: %u\r\n"

			"L1 code linesize: %u bytes\r\n"
			"L1 code lines per tag: %u\r\n"
			"L1 code associativity: %u\r\n"
			"L1 code size: %u KB\r\n"

			"L1 data linesize: %u bytes\r\n"
			"L1 data lines per tag: %u\r\n"
			"L1 data associativity: %u\r\n"
			"L1 data size: %u KB\r\n"

			"L2 data linesize: %u bytes\r\n"
			"L2 data lines per tag: %u\r\n"
			"L2 data associativity: %u\r\n"
			"L2 data size: %u KB\r\n"

			"L3 data linesize: %u bytes\r\n"
			"L3 data lines per tag: %u\r\n"
			"L3 data associativity: %u\r\n"
			"L3 data size: %u KB\r\n\r\n",

			pCPUInfo->L1CodeLargeTLB.entries,
			pCPUInfo->L1CodeLargeTLB.associativity,
			pCPUInfo->L1DataLargeTLB.entries,
			pCPUInfo->L1DataLargeTLB.associativity,

			pCPUInfo->L1CodeTLB.entries,
			pCPUInfo->L1CodeTLB.associativity,
			pCPUInfo->L1DataTLB.entries,
			pCPUInfo->L1DataTLB.associativity,

			pCPUInfo->L2DataLargeTLB.entries,
			pCPUInfo->L2DataLargeTLB.associativity,
			pCPUInfo->L2DataTLB.entries,
			pCPUInfo->L2DataTLB.associativity,

			pCPUInfo->L1Code.lineSize,
			pCPUInfo->L1Code.linesPerTag,
			pCPUInfo->L1Code.associativity,
			pCPUInfo->L1Code.size,

			pCPUInfo->L1Data.lineSize,
			pCPUInfo->L1Data.linesPerTag,
			pCPUInfo->L1Data.associativity,
			pCPUInfo->L1Data.size,

			pCPUInfo->L2Data.lineSize,
			pCPUInfo->L2Data.linesPerTag,
			pCPUInfo->L2Data.associativity,
			pCPUInfo->L2Data.size,

			pCPUInfo->L3Data.lineSize,
			pCPUInfo->L3Data.linesPerTag,
			pCPUInfo->L3Data.associativity,
			pCPUInfo->L3Data.size );
	printf("%s\n", buf);
}

