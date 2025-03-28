// ============================================================================
// TIMER FUNCTIONS
// Author: Yuhao Zhu, University of Rochester
// Full code avaliable at https://github.com/horizon-research/rtnn
#include "Timing.h"

std::unordered_map<int, AverageTime> Timing::m_averageTimes;
std::stack<TimingHelper> Timing::m_timingStack;
bool Timing::m_dontPrintTimes = false;
unsigned int Timing::m_startCounter = 0;
unsigned int Timing::m_stopCounter = 0;
std::string Timing::timingOutput = "";
std::string Timing::timingHeader = "";
