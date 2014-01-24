//
//  TimeUtils.cpp
//  GeometricMultiResolutionAnalysis
//
//  Created by Mauro Maggioni on 8/9/12.
//  Copyright (c) 2012 Mauro Maggioni. All rights reserved.
//

#include "TimeUtils.h"

using namespace std;


TimeList::~TimeList() {
    // times is destroyed automatically and so are the objects in the list
}

void TimeList::startClock( string Tag )  {
    uint64_t clockstart = mach_absolute_time();
    bool found = false;

    for(list<TimeToken>::iterator list_iter = times.begin(); list_iter != times.end(); list_iter++) {
        if( Tag.compare( list_iter->Tag )==0 ) {
            list_iter->clockstart = clockstart;
            found = true;
            break;
        }
    }
    if( !found )    {
        TimeToken newTimeToken;
        newTimeToken.clockstart = clockstart;
        newTimeToken.Tag.assign( Tag );
        newTimeToken.sec = 0.0;
        
        times.push_back(newTimeToken);
    }
}


double TimeList::endClock( string Tag ) {
    uint64_t clockend = mach_absolute_time();

    for(list<TimeToken>::iterator list_iter = times.begin(); list_iter != times.end(); list_iter++) {
        if( Tag.compare( list_iter->Tag )==0 ) {
            list_iter->sec = list_iter->sec + subtractTimes( clockend, list_iter->clockstart);
            return list_iter->sec;
        }
    }
    
    return 0.0;
}


ostream &operator<<( ostream &out, TimeList &timeList ) {
    for(list<TimeToken>::iterator list_iter = timeList.times.begin(); list_iter != timeList.times.end(); list_iter++) {
        cout << endl << list_iter->Tag << ":" << list_iter->sec;
    }
    
    return out;
}


// Raw mach_absolute_times going in, difference in seconds out
double subtractTimes( uint64_t endTime, uint64_t startTime )
{
    uint64_t difference = endTime - startTime;
    static double conversion = 0.0;
    
    if( conversion == 0.0 )
    {
        mach_timebase_info_data_t info;
        kern_return_t err = mach_timebase_info( &info );
        
        //Convert the timebase into seconds
        if( err == 0  )
            conversion = 1e-9 * (double) info.numer / (double) info.denom;
    }
    
    return conversion * (double) difference;
}
