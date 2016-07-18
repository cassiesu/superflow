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

#ifndef MEASURE_H
#define MEASURE_H

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

/* ==================== GNU C and possibly other UNIX compilers ===================== */
#ifndef WINK

#if defined(__GNUC__) || defined(__linux__)
#define VOLATILE __volatile__
#define ASM __asm__
#else
/* if we're neither compiling with gcc or under linux, we can hope
 * the following lines work, they probably won't */
#define ASM asm
#define VOLATILE
#endif

#define myInt64 unsigned long long
#define INTK unsigned int

/* ======================== WINK ======================= */
#else

#define myInt64 signed __int64
#define INTK unsigned __intK

#endif

#define COUNTER_LO(a) ((a).intK.lo)
#define COUNTER_HI(a) ((a).intK.hi)
#define COUNTER_VAL(a) ((a).int64)

#define COUNTER(a) \
		((unsigned long long)COUNTER_VAL(a))

#define COUNTER_DIFF(a,b) \
		(COUNTER(a)-COUNTER(b))

/* ==================== GNU C and possibly other UNIX compilers ===================== */
#ifndef WINK

typedef union
{       myInt64 int64;
struct {INTK lo, hi;} intK;
} tsc_counter;

#if defined(__ia64__)
#if defined(__INTEL_COMPILER)
#define RDTSC(tsc) (tsc).int64=__getReg(3116)
#else
#define RDTSC(tsc) ASM VOLATILE ("mov %0=ar.itc" : "=r" ((tsc).int64) )
#endif

#define CPUID() do{/*No need for serialization on Itanium*/}while(0)
#else
#define RDTSC(cpu_c) \
		ASM VOLATILE ("rdtsc" : "=a" ((cpu_c).intK.lo), "=d"((cpu_c).intK.hi))
#define CPUID() \
		ASM VOLATILE ("cpuid" : : "a" (0) : "bx", "cx", "dx" )
#endif

/* ======================== WINK ======================= */
#else

typedef union
{       myInt64 int64;
struct {INTK lo, hi;} intK;
} tsc_counter;

#define RDTSC(cpu_c)   \
		{       __asm rdtsc    \
	__asm mov (cpu_c).intK.lo,eax  \
	__asm mov (cpu_c).intK.hi,edx  \
		}

#define CPUID() \
		{ \
	__asm mov eax, 0 \
	__asm cpuid \
		}

#endif

#ifndef MAX_NUM_MEASUREMENTS
	#define MAX_NUM_MEASUREMENTS 100
#endif

extern myInt64 meas_values[MAX_NUM_MEASUREMENTS];
extern int meas_i;
extern myInt64 meas_start;
extern myInt64 meas_overhead;

/********************************************************************************/
// These can be used if we want to access the values of the perf. counters from our code

__attribute__((noinline)) void init_tsc();
__attribute__((noinline)) myInt64 start_tsc(void);
__attribute__((noinline)) myInt64 stop_tsc(myInt64 start);
__attribute__((noinline)) myInt64 get_tsc_overhead(void);

/********************************************************************************/
// These can be used if we don't care about the values of the perf. conters
// in our code and we just want the measurements to be written in measurements.txt

__attribute__((noinline)) void measurement_start(void);
__attribute__((noinline)) void measurement_stop(void);
__attribute__((noinline)) myInt64 get_measurement_overhead(void);
__attribute__((noinline)) void measurement_init(void);
__attribute__((noinline)) void measurement_finish(myInt64 * meas_values);

#endif
