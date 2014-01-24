//
//  TimeUtils.h
//  GeometricMultiResolutionAnalysis
//
//  Created by Mauro Maggioni on 8/9/12.
//  Copyright (c) 2012 Mauro Maggioni. All rights reserved.
//

#ifndef __GeometricMultiResolutionAnalysis__TimeUtils__
#define __GeometricMultiResolutionAnalysis__TimeUtils__

#include <iostream>
#include <list.h>
#include <mach/mach_time.h>
#include <time.h>
#include <string.h>

using namespace std;

typedef struct {
    string      Tag;
    double      sec;
    uint64_t    clockstart;
} TimeToken;

class TimeList {
    friend ostream &operator<<( ostream &out, TimeList &timeList );
private:
    list<TimeToken> times;
public:
    void startClock( string Tag );                                          // Restart a clock with given tag
    double endClock( string Tag );                                          // Stops a clock with given tag, and adds elapsed time since the last startClock to the corresponding <sec> field
    
    ~TimeList();
    
};

ostream &operator<<( ostream &out, TimeList &timeList );



double subtractTimes( uint64_t endTime, uint64_t startTime );

#endif /* defined(__GeometricMultiResolutionAnalysis__TimeUtils__) */
