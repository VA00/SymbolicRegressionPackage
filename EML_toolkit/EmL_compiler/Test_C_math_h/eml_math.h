#ifndef EML_MATH_H
#define EML_MATH_H

#include <complex.h>

static inline double complex eml(double complex x, double complex y) {
    return cexp(x) - clog(y);
}

static inline double complex eml_f(double complex x) {
    return eml(eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),eml(eml(1,eml(eml(1,eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),eml(eml(1,eml(eml(1,eml(eml(1,eml(eml(1,1),1)),eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),eml(1,1)),1))),1)),1)),1)),1)),1)),1)),1)),1))),1)),eml(eml(eml(1,eml(eml(1,eml(1,eml(eml(1,1),1))),1)),eml(eml(1,eml(eml(1,eml(1,eml(eml(1,x),1))),1)),1)),1)),1),1);
}

#endif
