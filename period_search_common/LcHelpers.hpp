#pragma once

#ifndef LCHELPERS_H
#define LCHELPERS_H

int PrepareLcData(struct globals& gl, const char* filename);
void MakeConvexityRegularization(struct globals& gl);

#endif
