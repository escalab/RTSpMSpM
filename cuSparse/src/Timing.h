// ============================================================================
// TIMER FUNCTIONS
// Author: Yuhao Zhu, University of Rochester
// Full code avaliable at https://github.com/horizon-research/rtnn
#ifndef __TIMING_H__
#define __TIMING_H__

#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline))
#endif

#include <iostream>
#include <stack>
#include <unordered_map>

#include <chrono>

struct TimingHelper
{
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::string name;
};

struct AverageTime
{
	double totalTime;
	unsigned int counter;
	std::string name;
};

class Timing
{
public:
	static bool m_dontPrintTimes;
	static unsigned int m_startCounter;
	static unsigned int m_stopCounter;
	static std::stack<TimingHelper> m_timingStack;
	static std::unordered_map<int, AverageTime> m_averageTimes;
	static std::string timingOutput;
	static std::string timingHeader;


	static void reset()
	{
		while (!m_timingStack.empty())
			m_timingStack.pop();
		m_averageTimes.clear();
		m_startCounter = 0;
		m_stopCounter = 0;
	}

	FORCE_INLINE static void startTiming(const std::string& name = std::string(""))
	{
		TimingHelper h;
		h.start = std::chrono::high_resolution_clock::now();
		h.name = name;
		Timing::m_timingStack.push(h);
		Timing::m_startCounter++;
	}

	FORCE_INLINE static double stopTiming(bool print = true)
	{
		if (!Timing::m_timingStack.empty())
		{
			Timing::m_stopCounter++;
			std::chrono::time_point<std::chrono::high_resolution_clock> stop = std::chrono::high_resolution_clock::now();
			TimingHelper h = Timing::m_timingStack.top();
			Timing::m_timingStack.pop();
			std::chrono::duration<double> elapsed_seconds = stop - h.start;
			double t = elapsed_seconds.count() * 1000.0;

			if (print)
			{
				Timing::timingHeader += h.name + ", ";
				Timing::timingOutput += std::to_string(t); //+ ", ";
			}
			return t;
		}
		return 0;
	}

	FORCE_INLINE static void flushTimer()
	{
		// std::cout << timingHeader << "\n" << std::flush;
		std::cout << timingOutput << std::flush;
		Timing::timingOutput.clear();
	}
};


#endif
